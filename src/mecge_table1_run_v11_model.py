"""Register v11 only for this entry point, then use the unmodified official runner."""

import models.lstm_v11  # noqa: F401
from mecge_table1_run_official_local_model import main


if __name__ == "__main__":
    main()
