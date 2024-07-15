import pandas as pd
from typing import Literal

durationType = pd.Timedelta | Literal["today", "yesterday"] | str
