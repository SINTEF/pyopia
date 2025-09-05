import json
import numpy as np
from pydantic import BaseModel, ConfigDict


class Metadata(BaseModel):
    """Default global metadata values stored in the PyOPIA stats netcdf file."""

    title: str = "NOT_SPECIFIED"
    project_name: str = "NOT_SPECIFIED"
    instrument: str = "NOT_SPECIFIED"
    seavox_instrument_identifier: str = "NOT_SPECIFIED"
    longitude: float = np.nan
    latitude: float = np.nan
    station: str = "NOT_SPECIFIED"
    creator_name: str = "NOT_SPECIFIED"
    creator_email: str = "NOT_SPECIFIED"
    creator_url: str = "NOT_SPECIFIED"
    institution: str = "NOT_SPECIFIED"
    license: str = "CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0/"

    # Allow extra metadata values from user
    model_config = ConfigDict(extra="allow")


if __name__ == "__main__":
    meta = Metadata()
    print(meta.json())

    meta_json = meta.model_dump()
    meta_json["extra_field"] = "My extra value!"
    meta = Metadata(**meta_json)
    assert meta.__pydantic_extra__ == {"extra_field": "My extra value!"}
    assert meta.extra_field == "My extra value!"
    print(meta)
    print(meta.json())

    with open("metadata_text.json", "w") as fh:
        json.dump(meta.model_dump(), fh, indent=4)

    with open("metadata_text.json", "r") as fh:
        meta_dict_in = json.load(fh)

    meta_in = Metadata(**meta_dict_in)

    print(meta_in)
