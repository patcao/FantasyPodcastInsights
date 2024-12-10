from datetime import datetime
from typing import TypeAlias, Union

import pandas as pd

DateLike: TypeAlias = Union[str, pd.Timestamp, datetime]
