<p align="center">
    <img height="238" src="./docs/images/logo-1.png" alt="ibat_logo">
</p>

<p align="center">
    <img src="https://img.shields.io/badge/-Project-blue" alt="project">
    <img src="https://img.shields.io/badge/-Research-yellowgreen" alt="research">
</p>
<p align="center">
    <img src="https://img.shields.io/badge/Bus%20arrival%20time%20prediction-8A2BE2" alt="batpre">
    <img src="https://img.shields.io/badge/Hybrid%20batch%20processing-8A2BE2" alt="hbp">
    <img src="https://img.shields.io/badge/Concept%20drift%20handling-8A2BE2" alt="cdrift">
    <img src="https://img.shields.io/badge/Active%20strategy-8A2BE2" alt="active">
    <img src="https://img.shields.io/badge/Incremental%20online%20learning-8A2BE2" alt="iol">
</p>

<p align="center">
    <b>iBAT</b> is a Python framework to enhance the real-time bus arrival time prediction in heterogenous traffic condition by addressing concept drift.
</p>

## Description

<p align="justify">
    This project aims to enhance the precision of real-time bus arrival time prediction models, particularly in the face of heterogeneous traffic conditions and dynamic external factors. The challenge lies in dealing with 'concept drift', which occurs due to evolving conditions like weather fluctuations, traffic variations, industrial impacts, road management changes, and unforeseen accidents. These factors contribute to a continuous shift in the patterns of bus arrival time data. To tackle this issue, we propose an innovative approach that involves the real-time detection of concept drift in the time series data. Once detected, our system quickly adapts to the new patterns by incorporating incremental online learning techniques into the existing state-of-the-art prediction model. This allows our model to efficiently learn from the evolving data within a short timeframe, resulting in more accurate and reliable bus arrival time predictions, ultimately benefiting both commuters and transportation services.
</p>

## Architecture

<p align="center">
    <img src="./docs/images/archi-dig.png" alt="archi-dig">
</p>

## 🛠 Installation

`iBAT` is intended to work with **Python 3.9 and above**. Installation can be done with `pip`:

```sh
pip install ibat
```

There are [wheels available](https://pypi.org/project/ibat/#files) for Linux, MacOS, and Windows. This means you most probably won't have to build iBAT from source.

You can install the latest development version from GitHub as so:

```sh
pip install git+https://github.com/aaivu/ibat --upgrade
pip install git+ssh://git@github.com/aaivu/ibat.git --upgrade  # using SSH
```

[//]: # (This method requires having Cython and Rust installed on your machine.)

## ⚡️ Quickstart

1. To run the pipeline, users can use their datasets or existing datasets from `ibat.datasets`.
```py
from ibat.datasets import BUS_654_DWELL_TIMES, BUS_654_RUNNING_TIMES
```

2. `concept_drift_detector` can be used to detect concept drift. User has a freedom to choose any strategy.
```py
from ibat.concept_drift_detector import CDD
from ibat.concept_drift_detector.strategies import Adwin

# You should define model, ni_x, and ni_y.

cdd = CDD(strategy=Adwin(delta=0.2))
is_detected = cdd.is_concept_drift_detected(model, ni_x, ni_y)
print(is_detected)
```

Output:
```
>>>> True
```

3. `pipeline` can be used to run experiments with the all the steps by simply running one function with some parameters.
```py
from datetime import datetime

from ibat.concept_drift_detector.strategies import DDM
from ibat.datasets import BUS_654_FEATURES_ENCODED_DWELL_TIMES
from ibat.pipeline import run_dt_exp


def datetime_from_string(datetime_string: str) -> datetime:
    return datetime.strptime(datetime_string, "%Y-%m-%d")


if __name__ == "__main__":
    cdd_strategy = DDM(
        warning_level=0.1,
        drift_level=1.5,
        min_num_instances=1,
    )
    run_dt_exp(
        dt_df=BUS_654_FEATURES_ENCODED_DWELL_TIMES.dataframe,
        hist_start=datetime_from_string("2021-10-01"),
        hist_end=datetime_from_string("2022-02-01"),
        stream_start=datetime_from_string("2022-02-01"),
        stream_end=datetime_from_string("2022-11-01"),
        interval_min=60 * 2,
        chunk_size=100,
        active_strategy=True,
        is_buffer_enabled=False,
        cdd_strategy=cdd_strategy,
        incremental_learning=True,
        output_parent_dir="./demo",
        label="demo-dt-exp-for-hbp",
    )
```

## 🤝 Contributors

<a href = "https://github.com/aaivu/ibat/graphs/contributors">

  <img src = "https://contrib.rocks/image?repo=aaivu/ibat" alt="contributors"/>

</a>

Made with [contributors-img](https://contrib.rocks).

[//]: # (1. [Kajanan Selvanesan]&#40;https://github.com/kajanan1212&#41;)

[//]: # (2. [Kesavi Aravinthan]&#40;https://github.com/KesaviAravinthan&#41; )

[//]: # (3. @3)

[//]: # (## 💬 Citation)

## 📝 License

`iBAT` is free and open-source software licensed under the [Apache License 2.0](https://github.com/aaivu/aaivu-introduction/blob/master/LICENSE).

## Code of Conduct

You can find our [code of conduct document here](https://github.com/aaivu/aaivu-introduction/blob/master/docs/code_of_conduct.md).
