import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import jellyfish
import numpy as np
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tqdm import tqdm

from src.player_utils import PlayerUtil, normalize_name
from src.utils import combine_chunks_with_overlap, get_repo_root


class PlayerNER:
    class PlayerList(BaseModel):
        players: list[str]

    NICKNAME_MAP = {
        "na'shon hylands": ["bones"],
        "stephen vurry": ["steph"],
        "lenron james": ["bron"],
        "kevin furant": ["kd"],
        "cameron johnson": ["cam johnson"],
        "cam thomas": ["cam thomas"],
        "herbert jones": ["herb jones"],
    }

    def __init__(self, model: str = "gpt-4o-mini"):
        if model == "gpt-4o-mini":
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        elif model == "gemini-1.5-flash":
            from langchain_google_vertexai import ChatVertexAI

            self.llm = ChatVertexAI(model="gemini-1.5-flash", temperature=0)
        else:
            raise RuntimeError(f"Model {model} not supported")
        self.player_util = PlayerUtil()
        self.all_players = self.player_util.get_all_players()
        self.all_players = self.all_players.assign(
            name_metaphone=self.all_players["personName"].apply(
                lambda name: jellyfish.metaphone(name)
            )
        )

        system_prompt = """You are a specialized NBA name entity recognizer. Identify all NBA players, correcting any misspellings or nicknames to the correct legal name of the player.

Requirements:
1. Identify all NBA players active in the 2023-24 season that are mentioned. 
2. Use context around the player mention to disambiguate which player is being referenced.
3. Do not include NBA team names, general managers, or any non players.
4. Use players legal names in the output, but replace any non standard english alphabet letters with the english alphabet.

You are given a mapping from the correct name of the player to their nickname to help with disambiguation:
{{
    "na'shon hylands": "bones",
    "dtephen vurry": "steph",
    "lenron james": "bron",
    "kevin furant": "kd",
    "cameron johnson: "cam johnson"
    "cam thomas": "cam thomas"
    "herbert jones": "herb jones"
}}

You should disambiguate player names using in the following order:
1. Context around the player mention like team names, names of teammates, or other player names.
2. Metaphone matching of player names to correct for misspellings or partial matches.
3. Nickname mapping to correct for common nicknames of players.

{format_instructions}

Context:
{text}
"""

        parser = PydanticOutputParser(pydantic_object=PlayerNER.PlayerList)
        prompt_template = PromptTemplate(
            template=system_prompt,
            input_variables=["text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        self.pipeline = prompt_template | self.llm | parser

    def _correct_player_nicknames(self, players: list[str]) -> list[str]:
        """Correct player nicknames to the correct legal name of the player."""
        corrected_player_names = []

        for player in players:
            for legal_name, nicknames in self.NICKNAME_MAP.items():
                if player in nicknames:
                    corrected_player_names.append(legal_name)
                    break
            else:
                corrected_player_names.append(player)

        return corrected_player_names

    def _correct_player_names_metaphone(self, players: list[str]) -> list[str]:
        """Correct player names based on metaphone matching (e.g., last names or partial matches)."""
        corrected_player_names = []

        suffixes = ["jr", "sr", "ii", "iii", "iv", "v", "vi", "vii"]
        players_df = self.all_players.assign(
            personNameNoSuffix=self.all_players["personName"].apply(
                lambda name: " ".join(
                    [word for word in name.split() if word.lower() not in suffixes]
                )
            )
        )
        players_df = players_df.assign(
            noSuffixMetaphone=players_df["personNameNoSuffix"].apply(
                lambda name: jellyfish.metaphone(name)
            )
        )

        for player in players:
            player_metaphone = jellyfish.metaphone(player)
            player_matches = players_df[
                (players_df["name_metaphone"] == player_metaphone)
                | (players_df["noSuffixMetaphone"] == player_metaphone)
            ]

            if len(player_matches) == 1:
                player_name = player_matches["personName"].values[0]
                corrected_player_names.append(player_name)
            else:
                corrected_player_names.append(player)

        return corrected_player_names

    def _correct_player_names_prefix(self, players: list[str]) -> list[str]:
        """Correct player names based on suffix matching (e.g., last names or partial matches).

        Args:
            players (list[str]): List of player names to correct.

        Returns:
            list[str]: List of corrected player names.
        """
        corrected_player_names = []

        for player in players:
            player_matches = self.all_players[
                self.all_players["personName"].str.startswith(player, na=False)
            ]

            if len(player_matches) == 1:
                player_name = player_matches["personName"].values[0]
                corrected_player_names.append(player_name)
            else:
                corrected_player_names.append(player)

        return corrected_player_names

    def correct_players(self, players: list[str]) -> list[str]:
        players = self._correct_player_nicknames(players)
        players = self._correct_player_names_prefix(players)
        players = self._correct_player_names_metaphone(players)

        return players

    def extract_all_players(self, text: str) -> list[str]:
        response = self.pipeline.invoke({"text": text})
        return [normalize_name(player) for player in response.players]


class FaissContainer:
    """
    A wrapper class for managing a FAISS index with metadata filtering
    Also caches of query embeddings as well as writing embeddings of text chunks to disk

    Attributes:
        index_name (str): Unique name of the FAISS index file for saving and loading.
        embeddings_model (Embeddings): A model to generate embeddings for text and queries.
        text_chunks (list[str]): List of text chunks to index.
        query_cache (dict): A cache for storing query embeddings to avoid recomputation.
        index (faiss.IndexFlatL2): The FAISS index object.
    """

    DEBUG = True

    def __init__(self, index_name: str, text_chunks: list[str], embeddings_model):
        self.index_name = index_name
        self.player_ner = PlayerNER()
        self.embeddings_model = embeddings_model
        self.text_chunks = text_chunks

        # File paths for pickled data
        self.directory = get_repo_root() / "data" / "faiss_indices" / index_name
        self.directory.mkdir(parents=True, exist_ok=True)

        self.chunk_embeddings_file = self.directory / "chunk_embeddings.pkl"
        self.player_chunk_mapping_file = self.directory / "player_chunk_mapping.pkl"

        self.chunk_embeddings = self._load_or_compute(
            self.chunk_embeddings_file,
            lambda: self.embeddings_model.embed_documents(text_chunks),
        )
        self.player_chunk_mapping = self._load_or_compute(
            self.player_chunk_mapping_file, lambda: self._annotate_chunks(text_chunks)
        )

        # Cache for query embeddings
        self.query_cache = {}

    def _load_or_compute(self, file_path: Path, compute_fn):
        if file_path.exists():
            if self.DEBUG:
                print(f"Loading from disk: {file_path.name}")

            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            data = compute_fn()
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            return data

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

    def _annotate_chunks(self, text_chunks: list[str]) -> Dict[str, list[int]]:
        player_chunk_mapping = {}

        for idx, chunk in enumerate(text_chunks):
            players = self.player_ner.extract_all_players(chunk)
            players = [self.player_ner.correct_players(n) for n in players]
            for player in players:
                if player not in player_chunk_mapping:
                    player_chunk_mapping[player] = []
                player_chunk_mapping[player].append(idx)

        return player_chunk_mapping

    def _create_faiss_index(self, embeddings) -> faiss.IndexFlatL2:
        """
        Create a new FAISS index using the provided text chunks and embedding model.

        Returns:
            faiss.IndexFlatL2: The created FAISS index.
        """
        # Convert embeddings list to NumPy array with dtype float32
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_array.shape[1]

        # Create the FAISS index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        return index

    def retrieve_relevant_indices(
        self, player: str, query: Optional[str], top_k: int = 3
    ) -> list[str]:
        player = normalize_name(player)
        if player not in self.player_chunk_mapping:
            return None

        chunk_indices = self.player_chunk_mapping[player]

        if len(chunk_indices) > top_k:
            relevant_embeddings = [self.chunk_embeddings[i] for i in chunk_indices]
            index = self._create_faiss_index(relevant_embeddings)

            query_embedding = self.embeddings_model.embed_query(query)
            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

            distances, indices = index.search(query_embedding, top_k)
            return distances[0], indices[0]
        else:
            return [0] * len(chunk_indices), chunk_indices


class PlayerAnalysis(BaseModel):
    personName: str
    increased_playing_time: float
    # mentions: int
    # trending_upwards: float


class PlayerAnalysisList(BaseModel):
    players: List[PlayerAnalysis]


class FaissFeatureExtractor:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        """
        Initialize the podcast summarizer with podcast text.

        Args:
            text (str): The podcast text.
            chunk_size (int): Maximum chunk size for splitting the text.
            chunk_overlap (int): Overlap between chunks to preserve context.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.embeddings_model = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        system_prompt = """You are a specialized NBA analyst. Analyze the following text each NBA player that is mentioned.
Requirements:
1. Analyze for the following players {player_list}.
2. Analyze the likelihood that each player will see increased minutes or playing time in upcoming games. Focus on discussions about rotation changes, injuries, coach decisions.
Give a value between -1 and 1, with 1 being very likely seeing increased playing time and -1 being very likely seeing decreased playing time, and a value of 0 if unsure or no relevant information is found.

Example 1:
Player: Xavier Tillman
Context: Xavier Tillman, obviously with the Steven Adams out for the season news, he got scooped up very quickly. In the opener, 17 points, 12 boards, 4 assists, 3 steals, 1 block.
Prediction: 0.8

Example 2:
Player: Derek Lively
Context: Derek Lively did not start the season opener but started the second half and played really well: 16 points, 10 rebounds, 1 steal, 1 block. I fully expect him to start moving forward unless Jason Kidd pulls some weird shenanigans.
Prediction: 0.5

Example 3:
Player: Santi Aldama
Context: Also, Santi Aldama, but we have yet to see him play. Santi is dealing with the ankle injury, hasn’t played, won’t play today. When healthy, he could be an interesting option.
Prediction: 0

Example 4:
Player: Mitchell Robinson
Context: Mitchell Robinson is confirmed to be out for the season, which eliminates his minutes entirely
Prediction: -1

Example 5:
Player: Josh Giddey
Context: Josh Giddey is mentioned as a drop candidate due to fit issues and inefficiencies, suggesting his minutes might be in jeopardy if his performance doesn’t improve.
Prediction: -0.8

{format_instructions}

Context:
{podcast_text}
"""
        parser = PydanticOutputParser(pydantic_object=PlayerAnalysisList)
        prompt_template = PromptTemplate(
            template=system_prompt,
            input_variables=["podcast_text", "player_list"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        self.pipeline = prompt_template | self.llm | parser

    def _extract_llm_feats_single_episode(
        self, episode_title: str, text: str
    ) -> pd.DataFrame:
        text = text[2 * self.chunk_size :]  # Skip the first 2 chunks
        text_chunks = self.text_splitter.split_text(text)

        # Create FAISS index for episode
        index_name = FaissContainer.generate_index_name(
            self, episode_title, self.chunk_size, self.chunk_overlap
        )
        faiss_index = FaissContainer(index_name, text_chunks, self.embeddings_model)

        # Get all chunks that contain relevant player context around "Increase or decrease in minutes and playing time"
        player_lst = list(faiss_index.player_chunk_mapping.keys())
        chunk_index_set = set()
        for player in player_lst:
            _, chunk_indices = faiss_index.retrieve_relevant_indices(
                player, "Increase or decrease in minutes and playing time"
            )
            chunk_index_set.update(chunk_indices)

        # Combine the relevant chunks with overlap
        combined_context = [text_chunks[i] for i in sorted(list(chunk_index_set))]
        combined_context = combine_chunks_with_overlap(
            combined_context, self.chunk_overlap
        )
        full_text = " ".join(combined_context)

        # Query the LLM pipeline to get player analysis features
        chunk_size = len(full_text) // 2
        text_chunks = [full_text[:chunk_size], full_text[chunk_size:]]

        struct_players = []
        for chunk in text_chunks:
            response = self.pipeline.invoke(
                {"podcast_text": chunk, "player_list": player_lst}
            )
            struct_players.extend(response.players)

        result = pd.DataFrame([player.dict() for player in struct_players])
        result["personName"] = result["personName"].apply(normalize_name)
        return result

    def extract_llm_feats(
        self, podcast_df: pd.DataFrame, aggregate: bool = True
    ) -> pd.DataFrame:
        df_list = []
        for row in tqdm(podcast_df.itertuples()):
            podcast_name = row.podcast_name
            episode_name = row.file_name
            podcast_date = pd.to_datetime(pd.to_datetime(row.publication_date).date())
            full_text = row.content

            single_episode_df = self._extract_llm_feats_single_episode(
                episode_name, full_text
            )
            single_episode_df["podcast_date"] = podcast_date
            single_episode_df["podcast_name"] = podcast_name

            df_list.append(single_episode_df)

        result = pd.concat(df_list)
        if aggregate:
            result = (
                result.groupby(["podcast_name", "personName", "podcast_date"])
                .agg(
                    {
                        "increased_playing_time": "mean",
                        # 'mentions': 'sum',
                        # 'trending_upwards': 'mean'
                    }
                )
                .reset_index()
                .sort_values(["podcast_date", "personName"])
            )

        return result


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
2. Analyze the likelihood that the player will see increased playing time in upcoming games. 
Give a value between -1 and 1, with 1 being very likely seeing increased playing time and -1 being very likely seeing decreased playing time.

Example 1:
Player: Xavier Tillman
Context: Xavier Tillman, obviously with the Steven Adams out for the season news, he got scooped up very quickly. In the opener, 17 points, 12 boards, 4 assists, 3 steals, 1 block.
Prediction: 0.8

Example 2:
Player: Derek Lively
Context: Derek Lively did not start the season opener but started the second half and played really well: 16 points, 10 rebounds, 1 steal, 1 block. I fully expect him to start moving forward unless Jason Kidd pulls some weird shenanigans.
Prediction: 0.5

Example 3:
Player: Santi Aldama
Context: Also, Santi Aldama, but we have yet to see him play. Santi is dealing with the ankle injury, hasn’t played, won’t play today. When healthy, he could be an interesting option.
Prediction: 0

Example 4:
Player: Mitchell Robinson
Context: Mitchell Robinson is confirmed to be out for the season, which eliminates his minutes entirely
Prediction: -1

Example 5:
Player: Josh Giddey
Context: Josh Giddey is mentioned as a drop candidate due to fit issues and inefficiencies, suggesting his minutes might be in jeopardy if his performance doesn’t improve.
Prediction: -0.8

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

    def extract_llm_feats(
        self, podcast_df: pd.DataFrame, aggregate: bool = True
    ) -> pd.DataFrame:
        # TDOO chunking in half means that player mentions will be duplicated in result dataframe
        df_list = []
        for row in tqdm(podcast_df.itertuples()):
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

        result = pd.concat(df_list)
        if aggregate:
            result = (
                result.groupby(["podcast_name", "personName", "podcast_date"])
                .agg(
                    {
                        "increased_playing_time": "mean",
                        # 'mentions': 'sum',
                        # 'trending_upwards': 'mean'
                    }
                )
                .reset_index()
                .sort_values(["podcast_date", "personName"])
            )

        return result

    # def summarize_player_mentions(self, player_name, top_k=5):
    #     """
    #     Summarize mentions of a specific player from the podcast text.

    #     Args:
    #         player_name (str): The player's name to search for.
    #         top_k (int): Number of relevant chunks to retrieve.

    #     Returns:
    #         str: Summary of discussions about the player.
    #     """
    #     query = f"Find all mentions and context about {player_name}."

    #     relevant_chunks = self.index.retrieve_relevant_chunks(query, top_k=top_k)
    #     if not relevant_chunks:
    #         return f"No mentions of {player_name} were found in the podcast."

    #     # Summarize the retrieved chunks using GPT
    #     context = "\n\n".join(relevant_chunks)
    #     prompt = (
    #         f"Summarize the discussions about {player_name} based on the following text:\n\n{context}\n\n"
    #         f"Provide a concise summary only about {player_name}. If a {player_name} is not mentioned, return 'Is not mentioned in the provided text'"
    #     )

    #     # TODO whether a player will see more or less playtime in the following game
    #     # TODO whether a player will be injured or not in the following game
    #     prompt = (
    #         f"Evaluate the injury risk of {player_name} based on the following text:\n\n{context}\n\n"
    #         f"On a scale of 1-10 where 10 is most likely to be out the next game, and 1 is likely to play, 5 is neutral. If a {player_name} is not mentioned, return None"
    #     )

    #     messages = [
    #         {
    #             "role": "system",
    #             "content": "You are a helpful assistant that summarizes discussions about NBA basketball palyers. ",
    #         },
    #         {"role": "user", "content": prompt},
    #     ]
    #     response = self.llm.invoke(prompt)
    #     return response.content

    # def summarize_player_mentions_batch(self, player_names, top_k=5):
    #     """
    #     Summarize mentions of multiple players in a single batch.

    #     Args:
    #         player_names (list[str]): List of player names to search for.
    #         top_k (int): Number of relevant chunks to retrieve for each player.

    #     Returns:
    #         dict: Player name as the key and their corresponding summary as the value.
    #     """
    #     # Retrieve relevant chunks for all players
    #     all_relevant_chunks = []
    #     for player_name in player_names:
    #         query = f"Find all mentions and context about {player_name}."
    #         relevant_chunks = self._retrieve_relevant_chunks(query, top_k=top_k)
    #         if not relevant_chunks:
    #             all_relevant_chunks.append(
    #                 (player_name, f"No mentions of {player_name} were found")
    #             )
    #         else:
    #             context = "\n\n".join(relevant_chunks)
    #             all_relevant_chunks.append((player_name, context))

    #     # Combine all player prompts into a single user message
    #     # combined_prompt = "Summarize the discussions about the following NBA players based on the provided text.'\n\n"
    #     combined_prompt = ""
    #     for player_name, context in all_relevant_chunks:
    #         combined_prompt += f"Player: {player_name}\nText: {context}\n\n"

    #     system_prompt = """You are a specialized NBA analyst.
    #     For each NBA player given, context about that player will follow.
    #     For context around each player that's given, perform the following analysis.

    #     Requirements:
    #     1. Identify all NBA players mentioned (current and former players)
    #     3. Count total mentions for each player
    #     4. Analyze whether a player is likely to see increased playing time in upcoming games
    #     4. Analyze whether a player is likely to outperform or trending upwards in upcoming games

    #     Give the output as a CSV with header.
    #     """
    #     # Create a messages list with separate entries for each player
    #     messages = [
    #         # {"role": "system", "content": "You are a helpful assistant that summarizes discussions about NBA basketball palyers. If a person is not mentioned, return 'Is not mentioned in the provided text'"},
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": combined_prompt},
    #     ]

    #     # Call the LLM once for the entire batch
    #     response = self.llm.invoke(messages)
    #     return response
