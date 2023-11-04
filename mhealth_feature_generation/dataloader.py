""" Import HealthKit data from different sources and standardize it into a single format.
Code for pulling cassandra data in ~/Code/dgc/Wellcomeleap/pipeline_april2023

"""
import pandas as pd
from typing import Literal
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm

DataFormatType = Literal["infer", "csv", "xml", "json"]


class DataLoader:
    def __init__(self):
        pass

    def loadData(self, path: str | Path) -> pd.DataFrame:
        """Loads HealthKit data from a file and returns a dataframe
        Assumes data is from 1 participant only
        Args:
            path (str | Path): path to data file

        Raises:
            NotImplementedError: Only supports xml or csv files
            ValueError: Only supports xml or csv files
            ValueError: Expected columns not in csv file

        Returns:
            pd.DataFrame: Formatted HealthKit Data
        """
        path = Path(path).expanduser().resolve()
        self.checkFileExists(path)

        name = Path(path).name
        if name.endswith(".csv"):
            data = self.loadCSV(path)
        elif name.endswith(".xml"):
            print("Loading XML data from", path)
            data = self.loadHealthKitXML(path)
            data = self.addLocalTime(data)
        elif name.endswith(".json"):
            raise NotImplementedError("JSON data format not yet supported")
        else:
            raise ValueError(f"Unknown file format: {path}")

        # If quantity and category separate, combine them to value col
        if ("body.quantity.value" in data.columns) or (
            "body.category.value" in data.columns
        ):
            data = data.rename(columns={"body.quantity.value": "value"})
            if "body.category.value" in data.columns:
                data.loc[
                    data["body.category.value"].notnull(), "value"
                ] = data.loc[
                    data["body.category.value"].notnull(),
                    "body.category.value",
                ]
                data = data.drop(columns=["body.category.value"])

        # If snake case, convert to camel case
        if (data.type.str.isupper()).any():
            data["type"] = data["type"].apply(self.snakeToCamelCase)
            data["type"] = data["type"].str.replace("Sdnn", "SDNN")

            sleep_vals = data["type"].str.contains("Sleep")
            if sleep_vals.any():
                data.loc[sleep_vals, "value"] = data.loc[
                    sleep_vals, "value"
                ].apply(self.snakeToCamelCase)

        data = data.rename(columns={"HKTimezone": "timezone"})
        # Check expected columns exist
        expected_columns = [
            "user_id",
            "local_start",
            "local_end",
            "timezone",
            "type",
        ]
        is_in = [c in data.columns for c in expected_columns]
        if not all(is_in):
            not_in = [c for c in expected_columns if c not in data.columns]
            raise ValueError(
                f"CSV file does not contain expected columns {not_in}: {path}"
            )

        return data

    @staticmethod
    def snakeToCamelCase(s: str) -> str:
        return "".join([z.capitalize() for z in s.split("_")])

    @staticmethod
    def checkFileExists(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Not a file: {path}")

    def loadCSV(self, path: Path | str) -> pd.DataFrame:
        """Load CSV data from a file and return a dataframe

        Args:
            path (Path | str): File path to healthkit CSV

        Raises:
            ValueError: File must be a CSV

        Returns:
            pd.DataFrame: HealthKit data
        """

        # Make sure path is a Path object
        path = Path(path).expanduser().resolve()
        self.checkFileExists(path)
        if not path.suffix == ".csv":
            raise ValueError(f"File is not a CSV: {path}")

        data = pd.read_csv(path, low_memory=False)
        data["local_start"] = pd.to_datetime(
            data["local_start"], format="ISO8601"
        )
        data["local_end"] = pd.to_datetime(data["local_end"], format="ISO8601")
        return data

    @staticmethod
    def loadXML(fp: Path):
        # create element tree object
        tree = ET.parse(fp.as_posix())
        return tree

    def loadHealthKitXML(
        self, path: Path, user_id: str = "anon"
    ) -> pd.DataFrame:
        hk_tree = self.loadXML(path)
        # for every health record, extract the attributes into a dictionary (columns). Then create a list (rows).
        root = hk_tree.getroot()
        # TODO: add parser for metadata and heart beat time series
        record_list = []
        for child in tqdm(root.iter("Record"), desc="Parsing HealthKit XML"):
            record = child.attrib
            for y in child.iter("MetadataEntry"):
                record[y.attrib["key"]] = y.attrib["value"]
            record_list.append(record)

        # create DataFrame from a list (rows) of dictionaries (columns)
        print("Converting Tree to DataFrame")
        data = pd.DataFrame(record_list)
        data.loc[data["type"] == "SleepAnalysis", "value"] = data.loc[
            data["type"] == "SleepAnalysis", "value"
        ].str.replace("HKCategoryValueSleepAnalysis", "")

        # proper type to dates
        for col in ["creationDate", "startDate", "endDate"]:
            data[col] = pd.to_datetime(data[col])

        # shorter observation names: use vectorized replace function
        data["type"] = data["type"].str.replace("HKQuantityTypeIdentifier", "")
        data["type"] = data["type"].str.replace("HKCategoryTypeIdentifier", "")
        data["user_id"] = user_id
        return data

    @staticmethod
    def addLocalTime(
        hk_data: pd.DataFrame, default_tz: str = "America/Los_Angeles"
    ) -> pd.DataFrame:
        """Add local time to HealthKit data
        Some HK records have timezone. HK data records are in UTC.
        Closest timezone is used for records without timezone.

        Args:
            hk_data (pd.DataFrame): HealthKit data loaded from XML
            default_tz (str, optional): Defaults to "America/Los_Angeles".

        Returns:
            pd.DataFrame: HK Data with timezone, local_start and local_end columns
        """
        timezone_data = hk_data[["startDate", "endDate", "HKTimeZone"]].copy()
        timezone_data["date_raw"] = timezone_data["startDate"].dt.date
        single_tz = (
            timezone_data[["startDate", "HKTimeZone"]]
            .drop_duplicates()
            .groupby("startDate")["HKTimeZone"]
            .first()
            .reset_index()
        )
        resamp_tz = (
            single_tz.set_index("startDate")["HKTimeZone"]
            .resample("1D", origin="start_day")
            .fillna("nearest")
            .reset_index()
        )
        resamp_tz["date_raw"] = resamp_tz["startDate"].dt.date
        hk_data["date_raw"] = hk_data["startDate"].dt.date
        hk_data = hk_data.drop(columns=["HKTimeZone"])
        hk_data = hk_data.merge(
            resamp_tz[["date_raw", "HKTimeZone"]], how="left"
        )
        hk_data["HKTimeZone"] = hk_data["HKTimeZone"].fillna(default_tz)

        hk_data["local_start"] = hk_data.apply(
            lambda row: pd.to_datetime(row["startDate"], utc=True)
            .tz_convert(row["HKTimeZone"])
            .tz_localize(None),
            axis=1,
        )
        hk_data["local_end"] = hk_data.apply(
            lambda row: pd.to_datetime(row["endDate"], utc=True)
            .tz_convert(row["HKTimeZone"])
            .tz_localize(None),
            axis=1,
        )
        hk_data = hk_data.rename(columns={"HKTimeZone": "timezone"})
        return hk_data

    # Function to get all OPTIMA studyhealthkit data
    def loadOPTIMAParticipantData(self, data_folder: Path, user_id: int) -> pd.DataFrame:
        hk_data_list = []
        sensor_folders = [
            f for f in data_folder.expanduser().iterdir() if f.is_dir()
        ]
        for sensor_folder in sensor_folders:
            sensor_path = Path(
                sensor_folder, f"{int(user_id)}-{sensor_folder.name}.csv"
            )
            if not sensor_path.exists():
                continue
            hk_data_list.append(self.loadData(sensor_path))
        if not hk_data_list:
            print(f"Skipping {user_id} due to no healthkit data")
            return pd.DataFrame()
        return pd.concat(hk_data_list)