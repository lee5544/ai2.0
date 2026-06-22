from __future__ import annotations

from pathlib import Path

import numpy as np
from nptdms import ChannelObject, GroupObject, RootObject, TdmsFile, TdmsWriter

from data_manager.tdms_read import materialize_seekable_tdms


def write_augmented_tdms(
    source_path: str | Path,
    output_path: str | Path,
    replacements: dict[tuple[str, str], np.ndarray],
) -> Path:
    """Write a TDMS copy with selected channel arrays replaced."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with materialize_seekable_tdms(source_path) as readable_path:
        with TdmsFile.open(readable_path) as original:
            objects: list[object] = [RootObject(original.properties)]
            for group in original.groups():
                objects.append(GroupObject(group.name, group.properties))
                for channel in group.channels():
                    data = replacements.get((group.name, channel.name))
                    if data is None:
                        data = channel[:]
                    objects.append(
                        ChannelObject(
                            group.name,
                            channel.name,
                            data,
                            properties=channel.properties,
                        )
                    )
            with TdmsWriter(output_path) as writer:
                writer.write_segment(objects)
    return output_path
