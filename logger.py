import os

class Logger:

    color_code_map = {
        'red': "\033[31m",
        'green': "\033[32m",
        'cyan': "\033[36m",
        'yellow': "\033[33m"
    }

    reset = "\033[0m"

    def __init__(self, debug_mode=False, logs_folder_path=""):
        self.debeg_mode = debug_mode
        self.debug_count = 0
        self.log_count = 0
        self.checkpoint_count = 0

        # Set up logs folder
        self.logs_folder = logs_folder_path
        os.makedirs(self.logs_folder, exist_ok=True)

        self.debug_log_file = os.path.join(self.logs_folder, "debug.logs")
        self.log_file = os.path.join(self.logs_folder, "logs.logs")
        self.checkpoint_file = os.path.join(self.logs_folder, "checkpoint.logs")

    def _log_colored_to_console(self, string: str, source, counter, color, end="\n"):
        color_code = self.color_code_map[color]
        print(f"{color_code}[{source}:{counter}] {self.reset}{string}", end=end)

    def _write_to_file(self, file_path, source, counter, string):
        with open(file_path, "a") as f:
            f.write(f"[{source}:{counter}] {string}\n")

    def debug(self, string: str, obj=None, end="\n"):
        if self.debeg_mode:
            counter_str = str(self.debug_count).rjust(3, '0')
            source = 'DEBUGGER'
            color = 'red'
            if obj is not None:
                string = f' {string}: {obj}'

            self._log_colored_to_console(string, source, counter_str, color, end=end)
            self._write_to_file(self.debug_log_file, source, counter_str, string)

            self.debug_count += 1

    def log(self, string: str, obj=None, end="\n"):
        counter_str = str(self.log_count).rjust(3, '0')
        source = 'LOGGER'
        color = 'green'
        if obj is not None:
            string = f' {string}: {obj}'

        self._log_colored_to_console(string, source, counter_str, color, end=end)
        self._write_to_file(self.log_file, source, counter_str, string)

        self.log_count += 1

    def checkpoint(self, string: str, end="\n"):
        counter_str = str(self.checkpoint_count).rjust(3, '0')
        source = 'CHECKPOINT'
        color = 'cyan'

        self._log_colored_to_console(string, source, counter_str, color, end=end)
        self._write_to_file(self.checkpoint_file, source, counter_str, string)

        self.checkpoint_count += 1
