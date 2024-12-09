import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
import numpy as np
from langchain.chat_models import ChatOpenAI
import os
import pickle
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from typing import List, Optional
import pandas as pd
from src.constants import DateLike
from src.player_utils import normalize_name


class FaissContainer:
    """
    A wrapper class for managing a FAISS index with caching of query embeddings
    and persistence of the index to disk.

    Attributes:
        index_name (str): Unique name of the FAISS index file for saving and loading.
        embeddings_model (Embeddings): A model to generate embeddings for text and queries.
        text_chunks (list[str]): List of text chunks to index.
        query_cache (dict): A cache for storing query embeddings to avoid recomputation.
        index (faiss.IndexFlatL2): The FAISS index object.
    """

    def __init__(self, index_name: str, text_chunks: list[str], embeddings_model):
        self.index_name = index_name
        self.embeddings_model = embeddings_model
        self.text_chunks = text_chunks

        # Cache for query embeddings
        self.query_cache = {}

        # TODO need to set the output directory of the indices

        # Load or create index
        index_exists = False
        if index_exists:
            self.index = self._load_faiss_index()
        else:
            self.index = self._create_faiss_index()
            self._save_faiss_index()

    @staticmethod
    def generate_index_name(
        cls, podcast_title: str, chunk_size: int, chunk_overlap: int
    ) -> str:
        """
        Generate a unique index name based on podcast title and chunking parameters.

        Parameters:
            podcast_title (str): Title of the podcast.
            chunk_size (int): Size of the text chunks.
            chunk_overlap (int): Overlap between consecutive text chunks.
        """
        return f"{podcast_title}_chunk{chunk_size}_overlap{chunk_overlap}"

    def _load_faiss_index(self) -> faiss.IndexFlatL2:
        """Load FAISS index from disk."""
        with open(f"{self.index_name}.pkl", "rb") as f:
            index_data = pickle.load(f)
        index = faiss.deserialize_index(index_data["index"])
        return index

    def _save_faiss_index(self) -> None:
        """Save FAISS index to disk."""
        index_data = {
            "index": faiss.serialize_index(self.index),
        }
        with open(f"{self.index_name}.pkl", "wb") as f:
            pickle.dump(index_data, f)

    def _create_faiss_index(self) -> faiss.IndexFlatL2:
        """
        Create a new FAISS index using the provided text chunks and embedding model.

        Returns:
            faiss.IndexFlatL2: The created FAISS index.
        """
        embeddings = self.embeddings_model.embed_documents(self.text_chunks)

        # Convert embeddings list to NumPy array with dtype float32
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_array.shape[1]

        # Create the FAISS index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        return index

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> list[str]:
        """
        Retrieve the most relevant text chunks for a given query.

        Parameters:
            query (str): Query string to search the index.
            top_k (int): Number of top relevant chunks to retrieve.

        Returns:
            list[str]: The top-k relevant text chunks.
        """

        # Check if query embedding is already cached
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.embeddings_model.embed_query(query)
            self.query_cache[query] = query_embedding

        query_embedding = self.embeddings_model.embed_query(query)

        # Convert to NumPy array and ensure 2D shape (1, d)
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # Search the FAISS index for the top_k nearest neighbors
        distances, indices = self.index.search(query_embedding, top_k)
        # TODO distance filter

        return [self.text_chunks[i] for i in indices[0]]


class FaissFeatureExtractor:
    def __init__(
        self, podcast_title: str, podcast_text: str, chunk_size=1000, chunk_overlap=200
    ):
        """
        Initialize the podcast summarizer with podcast text.

        Args:
            text (str): The podcast text.
            chunk_size (int): Maximum chunk size for splitting the text.
            chunk_overlap (int): Overlap between chunks to preserve context.
        """
        self.text = podcast_text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.chunks = self.text_splitter.split_text(podcast_text)
        self.embeddings_model = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.index_name = FaissContainer.generate_index_name(
            self, podcast_title, chunk_size, chunk_overlap
        )
        self.index = FaissContainer(self.index_name, self.chunks, self.embeddings_model)

    def summarize_player_mentions(self, player_name, top_k=5):
        """
        Summarize mentions of a specific player from the podcast text.

        Args:
            player_name (str): The player's name to search for.
            top_k (int): Number of relevant chunks to retrieve.

        Returns:
            str: Summary of discussions about the player.
        """
        query = f"Find all mentions and context about {player_name}."

        relevant_chunks = self.index.retrieve_relevant_chunks(query, top_k=top_k)
        if not relevant_chunks:
            return f"No mentions of {player_name} were found in the podcast."

        # Summarize the retrieved chunks using GPT
        context = "\n\n".join(relevant_chunks)
        prompt = (
            f"Summarize the discussions about {player_name} based on the following text:\n\n{context}\n\n"
            f"Provide a concise summary only about {player_name}. If a {player_name} is not mentioned, return 'Is not mentioned in the provided text'"
        )

        # TODO whether a player will see more or less playtime in the following game
        # TODO whether a player will be injured or not in the following game
        prompt = (
            f"Evaluate the injury risk of {player_name} based on the following text:\n\n{context}\n\n"
            f"On a scale of 1-10 where 10 is most likely to be out the next game, and 1 is likely to play, 5 is neutral. If a {player_name} is not mentioned, return None"
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes discussions about NBA basketball palyers. ",
            },
            {"role": "user", "content": prompt},
        ]
        response = self.llm.invoke(prompt)
        return response.content

    def summarize_player_mentions_batch(self, player_names, top_k=5):
        """
        Summarize mentions of multiple players in a single batch.

        Args:
            player_names (list[str]): List of player names to search for.
            top_k (int): Number of relevant chunks to retrieve for each player.

        Returns:
            dict: Player name as the key and their corresponding summary as the value.
        """
        # Retrieve relevant chunks for all players
        all_relevant_chunks = []
        for player_name in player_names:
            query = f"Find all mentions and context about {player_name}."
            relevant_chunks = self._retrieve_relevant_chunks(query, top_k=top_k)
            if not relevant_chunks:
                all_relevant_chunks.append(
                    (player_name, f"No mentions of {player_name} were found")
                )
            else:
                context = "\n\n".join(relevant_chunks)
                all_relevant_chunks.append((player_name, context))

        # Combine all player prompts into a single user message
        # combined_prompt = "Summarize the discussions about the following NBA players based on the provided text.'\n\n"
        combined_prompt = ""
        for player_name, context in all_relevant_chunks:
            combined_prompt += f"Player: {player_name}\nText: {context}\n\n"

        system_prompt = """You are a specialized NBA analyst. 
        For each NBA player given, context about that player will follow.
        For context around each player that's given, perform the following analysis.        

        Requirements:
        1. Identify all NBA players mentioned (current and former players)
        3. Count total mentions for each player
        4. Analyze whether a player is likely to see increased playing time in upcoming games
        4. Analyze whether a player is likely to outperform or trending upwards in upcoming games
        
        Give the output as a CSV with header.
        """
        # Create a messages list with separate entries for each player
        messages = [
            # {"role": "system", "content": "You are a helpful assistant that summarizes discussions about NBA basketball palyers. If a person is not mentioned, return 'Is not mentioned in the provided text'"},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_prompt},
        ]

        # Call the LLM once for the entire batch
        response = self.llm.invoke(messages)
        return response


class PlayerAnalysis(BaseModel):
    personName: str
    mentions: int
    increased_playing_time: float
    trending_upwards: float


class PlayerAnalysisList(BaseModel):
    players: List[PlayerAnalysis]


class PromptFeatureExtractor:
    def __init__(self, system_prompt: Optional[str] = None):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # system_prompt = """You are a specialized NBA analyst. Analyze the following text for NBA player mentions, including pronouns and nicknames.
        # Requirements:
        # 1. Identify all NBA players active in the 2023 season mentioned. Use their real names in the output.
        # 3. Count total mentions for each player
        # 4. Analyze whether a player is likely to see increased playing time in upcoming games
        # 5. Analyze whether a player is likely to outperform or trending upwards in upcoming games
        # {format_instructions}
        # Context:
        # {podcast_text}
        # """

        if system_prompt is None:
            system_prompt = """You are a specialized NBA analyst. Analyze the following text for NBA player mentions, including pronouns and nicknames.
Requirements:
1. Identify all NBA players active in the 2023 season mentioned. Use their real names in the output.
3. Count total mentions for each player
4. Analyze how likely a player is to see increased playing time in upcoming games. Give a value between 0 and 1, with 1 being definitely seeing increased playing time.
5. Analyze whether a player is likely to outperform or trending upwards in upcoming games. Give a value between 0 and 1, with 1 being definitely trending upwards.

{format_instructions}

Context:
{podcast_text}
"""

        parser = PydanticOutputParser(pydantic_object=PlayerAnalysisList)
        prompt_template = PromptTemplate(
            template=system_prompt,
            input_variables=["podcast_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        self.pipeline = prompt_template | self.llm | parser

    def extract_llm_feats(self, podcast_df) -> pd.DataFrame:
        # TDOO chunking in half means that player mentions will be duplicated in result dataframe

        df_list = []
        for row in podcast_df.itertuples():
            full_text = row.content
            podcast_date = pd.to_datetime(pd.to_datetime(row.publication_date).date())

            chunk_size = len(full_text) // 2
            text_chunks = [full_text[:chunk_size], full_text[chunk_size:]]

            struct_players = []
            for chunk in text_chunks:
                response = self.pipeline.invoke({"podcast_text": chunk})
                struct_players.extend(response.players)

            struct_df = pd.DataFrame([player.dict() for player in struct_players])
            # struct_df = struct_df.groupby('personName').mean().reset_index()
            struct_df["podcast_date"] = podcast_date
            struct_df["podcast_name"] = row.podcast_name
            struct_df["personName"] = struct_df["personName"].apply(normalize_name)

            df_list.append(struct_df)

        return pd.concat(df_list)
