from typing import Union, TypeAlias
from datetime import datetime
import pandas as pd

DateLike: TypeAlias = Union[str, pd.Timestamp, datetime]
