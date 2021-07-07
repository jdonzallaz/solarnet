import logging
import shutil
from pathlib import Path
from time import strptime
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from sunpy.net import Fido, attrs as a
from tqdm import tqdm

from solarnet.utils.dict import print_dict
from solarnet.utils.physics import class_to_flux
from solarnet.utils.target import flux_to_class_builder
from solarnet.utils.yaml import write_yaml

logger = logging.getLogger(__name__)


def make_dataset(parameters: dict):
    print_dict(parameters)

    destination = Path(parameters["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / parameters["filename"]
    dataset_path = Path(parameters["dataset_path"])
    relative_paths = parameters["relative_paths"]
    copy = parameters["copy"]

    # Write config
    write_yaml(destination / "sdo-dataset-config.yaml", parameters)

    channel = parameters["channel"]

    time_steps = [pd.Timedelta(t) for t in parameters["time-steps"]]
    flare_period = pd.Timedelta(parameters["flare-period"])
    sample_interval = pd.Timedelta(parameters["sample-interval"])
    search_image_period = pd.Timedelta(parameters["search-image-period"])
    target = parameters["target"]
    classification = "classes" in target

    # Splits
    splits = parameters["splits"]
    margin_between_splits = pd.Timedelta(parameters["margin-between-splits"])
    splits_separate_files = parameters["splits-separate-files"]
    if splits is None or len(splits) < 1:
        raise AttributeError("At least one split is required")
    split_dict = {
        strptime(part, "%b").tm_mon if isinstance(part, str) else part: split
        for split, parts in splits.items()
        for part in parts
    }

    first_known_datetime = pd.Timestamp("2010/05/13T00:00:00")
    last_known_datetime = pd.Timestamp("2018/12/31T23:59:59")

    datetime_start = first_known_datetime
    datetime_end = last_known_datetime

    if not dataset_path.exists():
        raise ValueError("Dataset not found at this location")

    logger.info("Downloading flares from HEK...")
    df = get_flares(datetime_start, datetime_end)

    print(df)

    # Samples
    samples: List[Dict[str, Union[List[str], float, pd.Timestamp, bool]]] = []

    time_offset = pd.DateOffset(seconds=sample_interval.total_seconds())
    intervals = pd.interval_range(datetime_start, datetime_end, freq=time_offset)

    logger.info("Searching corresponding image files in the dataset...")

    old_split = None
    split_date = None
    old_split_date = None
    for interval in intervals:
        try:
            images_paths, all_found = find_paths(
                dataset_path, interval.left, time_steps, channel, search_image_period, relative_paths
            )
        except FileNotFoundError as e:
            continue
        peak_flux = find_flare_peak_flux(df, interval.left, interval.left + flare_period)

        sample_datetime: pd.Timestamp = interval.left

        # Prepare the split: choose month or year + ignore samples in the "margin" between splits
        split = split_dict.get(
            sample_datetime.year,
            split_dict.get(sample_datetime.month, None),
        )

        if split != old_split:
            old_split_date = split_date
        old_split = split
        split_date = sample_datetime

        if old_split_date is not None and split_date - margin_between_splits < old_split_date:
            continue

        if split is None:
            continue

        samples.append(
            {
                "images_paths": images_paths,
                "peak_flux": peak_flux,
                "datetime": sample_datetime,
                "all_found": all_found,
                "split": split,
            }
        )

    samples_ls = [[*i["images_paths"], i["peak_flux"], i["datetime"], i["all_found"], i["split"]] for i in samples]

    columns = [f"path_{i}_before" for i in parameters["time-steps"]] + ["peak_flux", "datetime", "all_found", "split"]
    df = pd.DataFrame(samples_ls, columns=columns)

    # Map target to classes if necessary
    if classification:
        target_transform = flux_to_class_builder(target["classes"], return_names=True)
        df["peak_flux"] = df["peak_flux"].map(target_transform)

    if splits_separate_files:
        for split in splits:
            df_split = df[df["split"] == split]
            split_csv_path = csv_path.parent / f"{csv_path.stem}-{split}{csv_path.suffix}"
            df_split.to_csv(split_csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)

    if copy:
        paths = (
            path for sample in samples for path in sample["images_paths"] if sample["split"] != "ignore"
        )  # Generator of paths
        n_files = len(samples) * len(time_steps)
        copy_data(paths, dataset_path, destination, n_files)


def copy_data(paths: Iterable[str], dataset_path: Path, destination: Path, n_files: int):
    logger.info("Copy files to destination")
    count = 0
    for i in tqdm(paths, disable=logging.root.level > logging.INFO, total=n_files):
        from_ = dataset_path / i
        to_ = destination / i
        to_.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(from_, to_)
        count += 1
    logger.info(f"Copied {count} files to {destination}")


def time_ranges_generator(datetime: pd.Timestamp, range: int = 6, unit: str = "min"):
    """
    Generate times around a datetime with a given range.
    It takes the initial datetime and generate:
        datetime - {range}{unit}, datetime + {range}{unit}, datetime - 2*{range}{unit}, datetime + 2*{range}{unit}, ...

    :param datetime: The initial datetime
    :param range: The range for the jumps
    :param unit: The unit of the range
    :return: yield an infinite number of datetime with given range
    """

    current_range = range
    while True:
        td = pd.Timedelta(current_range, unit)
        yield datetime - td
        yield datetime + td
        current_range += range


def make_path(base_path, time, channel) -> Path:
    """
    Create path like:
    HMI20100501_1736_bz.npz
    AIA20180114_0242_0171.npz
    """

    year = str(time.year)
    month = f"{time.month:02d}"
    day = f"{time.day:02d}"
    hours = f"{time.hour:02d}"
    minutes = f"{time.minute:02d}"

    if isinstance(channel, str) and channel.lower() in ["bx", "by", "bz"]:
        instrument = "HMI"
        filename_channel = channel.lower()
    else:
        instrument = "AIA"
        filename_channel = f"{channel:04d}"

    return (
        base_path
        / str(channel)
        / year
        / month
        / day
        / f"{instrument}{year}{month}{day}_{hours}{minutes}_{filename_channel}.npz"
    )


def find_paths(
    base_path: Path,
    datetime: pd.Timestamp,
    time_steps: List[pd.Timedelta],
    channel: str,
    search_period: pd.Timedelta,
    relative_paths: bool,
) -> Tuple[List[Path], bool]:
    paths = []
    all_found = True

    for time_step in time_steps:
        datetime_before = datetime - time_step
        datetime_before = datetime_before.round("6min")

        path = make_path(base_path, datetime_before, channel)

        if not path.exists():
            all_found = False

            datetime_bottom_search_range = datetime_before - search_period
            datetime_top_search_range = datetime_before + search_period
            search_range = pd.Interval(datetime_bottom_search_range, datetime_top_search_range)

            gen = time_ranges_generator(datetime_before)
            while True:
                new_time = next(gen)
                if new_time not in search_range:
                    raise FileNotFoundError()
                path = make_path(base_path, new_time, channel)
                if path.exists():
                    break

        if relative_paths:
            path = path.relative_to(base_path)

        paths.append(path)

    return paths, all_found


def find_flare_peak_flux(df: pd.DataFrame, start, end) -> float:
    flares_df = df.loc[start:end]
    flux_values = flares_df["fl_goescls"].values

    if len(flux_values) == 0:
        return 1e-9

    return max(map(class_to_flux, flux_values))


def get_flares_cache_filename(datetime_start: pd.Timestamp, datetime_end: pd.Timestamp):
    """
    Return a corretly formatted filename for a parquet flares cache file, with dates (range) for reference.
    """

    return (
        f"hek_flares_{datetime_start.isoformat().replace(':', '-')}_"
        f"{datetime_end.isoformat().replace(':', '-')}.parquet"
    )


def get_flares_from_cache(datetime_start: pd.Timestamp, datetime_end: pd.Timestamp) -> Optional[pd.DataFrame]:
    """
    Check in the cache for data in the given dates range. Return None if nothing is found.
    """

    folder = Path.home() / ".solarnet" / "hek"
    filename = get_flares_cache_filename(datetime_start, datetime_end)
    path = folder / filename

    if not path.exists():
        return None

    logger.info("Loading flares from cache")

    try:
        df = pd.read_parquet(path)
    except Exception:
        logger.warning("Error while loading flares cache")
        return None

    return df


def write_flares_to_cache(df: pd.DataFrame, datetime_start: pd.Timestamp, datetime_end: pd.Timestamp):
    """
    Write the given dataframe to a .parquet file for caching. Uses the given dates range for reference.
    The cache is in the .solarnet/hek/ folder of the user home directory.
    """

    folder = Path.home() / ".solarnet" / "hek"
    folder.mkdir(parents=True, exist_ok=True)
    filename = get_flares_cache_filename(datetime_start, datetime_end)
    path = folder / filename

    logger.info("Writing flares to cache")

    try:
        df.to_parquet(path)
    except Exception:
        logger.warning("Error while writing flares to cache")


def get_flares(datetime_start: pd.Timestamp, datetime_end: pd.Timestamp):
    df = get_flares_from_cache(datetime_start, datetime_end)
    if df is not None:
        return df

    observatory = "GOES"
    instrument = "GOES"
    from_name = "SWPC"
    columns = [
        "frm_name",
        "obs_observatory",
        "obs_instrument",
        "ar_noaanum",
        "event_starttime",
        "event_endtime",
        "event_peaktime",
        "fl_goescls",
        "fl_peakflux",
    ]

    result = Fido.search(
        a.Time(datetime_start, datetime_end),
        a.hek.FL,
        a.hek.OBS.Observatory == observatory,
        a.hek.OBS.Instrument == instrument,
        a.hek.FRM.Name == from_name,
    )

    hek = result["hek"]
    hek.keep_columns(columns)
    hek = hek[hek["ar_noaanum"] != 0]
    df = hek.to_pandas()

    # Index by event_peaktime column and parse date
    df = df.astype({"event_peaktime": "datetime64[ns]"})
    df = df.set_index("event_peaktime")
    df = df.sort_index()

    write_flares_to_cache(df, datetime_start, datetime_end)

    return df
