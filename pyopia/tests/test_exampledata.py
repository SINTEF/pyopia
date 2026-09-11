import logging

import pyopia.exampledata


def test_progress_hook_logs_at_ten_percent_increments(caplog):
    hook = pyopia.exampledata._make_progress_hook("test.zip")
    block_size = 1024
    total_size = 10 * block_size  # 10 blocks = 10% per block

    with caplog.at_level(logging.INFO, logger=pyopia.exampledata.logger.name):
        for block_count in range(1, 11):
            hook(block_count, block_size, total_size)

    percents_logged = [int(r.message.rsplit(' ', 1)[-1].rstrip('%')) for r in caplog.records]
    assert percents_logged == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def test_progress_hook_does_not_log_between_increments(caplog):
    hook = pyopia.exampledata._make_progress_hook("test.zip")
    block_size = 1
    total_size = 1000  # each block is 0.1%, well under the 10% threshold

    with caplog.at_level(logging.INFO, logger=pyopia.exampledata.logger.name):
        for block_count in range(1, 50):
            hook(block_count, block_size, total_size)

    assert len(caplog.records) == 0


def test_progress_hook_ignores_unknown_total_size(caplog):
    hook = pyopia.exampledata._make_progress_hook("test.zip")

    with caplog.at_level(logging.INFO, logger=pyopia.exampledata.logger.name):
        hook(1, 1024, -1)

    assert len(caplog.records) == 0
