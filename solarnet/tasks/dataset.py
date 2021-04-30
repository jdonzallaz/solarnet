import logging
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from sunpy.net import Fido, attrs as a

from solarnet.utils.dict import print_dict
from solarnet.utils.physics import class_to_flux
from solarnet.utils.target import flux_to_class_builder
from solarnet.utils.yaml import write_yaml

logger = logging.getLogger(__name__)


def make_dataset(parameters: dict):
    print_dict(parameters)

    destination = Path(parameters["destination"])
    csv_path = destination / parameters["filename"]
    dataset_path = Path(parameters["dataset_path"])
    relative_paths = parameters["relative_paths"]

    # Write config
    write_yaml(destination / "sdo-dataset-config.yaml", parameters)

    channel = parameters["channel"]

    time_steps = [pd.Timedelta(t) for t in parameters["time-steps"]]
    flare_time_range = pd.Timedelta(parameters["flare-time-range"])
    search_image_time_range = pd.Timedelta(parameters["search-image-time-range"])
    target = parameters["target"]
    classification = "classes" in target

    splits = parameters["splits"]
    splits_separate_files = parameters["splits-separate-files"]
    if splits is None or len(splits) < 1:
        raise AttributeError("At least one split is required")
    print_dict(splits)
    year_to_split_dict = {year: split for split, years in splits.items() for year in years}

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

    time_offset = pd.DateOffset(seconds=flare_time_range.total_seconds())
    intervals = pd.interval_range(
        datetime_start,  # + max(time_steps),
        datetime_end,
        freq=time_offset
    )

    for interval in intervals:
        try:
            images_paths, all_found = find_paths(
                dataset_path, interval.left, time_steps, channel, search_image_time_range, relative_paths)
        except FileNotFoundError as e:
            # print("1 path not found", interval.left)
            continue
        peak_flux = find_flare_peak_flux(df, interval.left, interval.right)

        sample_datetime: pd.Timestamp = interval.left

        samples.append({
            "images_paths": images_paths,
            "peak_flux": peak_flux,
            "datetime": sample_datetime,
            "all_found": all_found,
            "split": year_to_split_dict.get(sample_datetime.year, None),
        })

    samples_ls = [[*i["images_paths"], i["peak_flux"], i["datetime"], i["all_found"], i["split"]] for i in samples]

    columns = [f"path_{i}_before" for i in parameters["time-steps"]] + ["peak_flux", "datetime", "all_found", "split"]
    df = pd.DataFrame(samples_ls, columns=columns)

    # Map target to classes if necessary
    if classification:
        target_transform = flux_to_class_builder(target["classes"], return_names=True)
        df["peak_flux"] = df["peak_flux"].map(target_transform)

    if splits_separate_files:
        for split in splits:
            df_split = df[df['split'] == split]
            split_csv_path = csv_path.parent / f"{csv_path.stem}-{split}{csv_path.suffix}"
            df_split.to_csv(split_csv_path, index=False)
        return

    df.to_csv(csv_path, index=False)


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
    year = str(time.year)
    month = f"{time.month:02d}"
    day = f"{time.day:02d}"
    hours = f"{time.hour:02d}"
    minutes = f"{time.minute:02d}"
    padded_channel = f"{channel:04d}"

    return base_path / str(channel) / year / month / day / \
           f"AIA{year}{month}{day}_{hours}{minutes}_{padded_channel}.npz"


def find_paths(
    base_path: Path,
    datetime: pd.Timestamp,
    time_steps: List[pd.Timedelta],
    channel: str,
    search_time_range: pd.Timedelta,
    relative_paths: bool,
) -> (List[Path], bool):
    paths = []
    all_found = True

    for time_step in time_steps:
        datetime_before = datetime - time_step
        datetime_before = datetime_before.round("6min")

        path = make_path(base_path, datetime_before, channel)

        if not path.exists():
            all_found = False

            datetime_bottom_search_range = datetime_before - search_time_range
            datetime_top_search_range = datetime_before + search_time_range
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


def get_flares(datetime_start, datetime_end):
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

    # TODO: add caching
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

    return df
