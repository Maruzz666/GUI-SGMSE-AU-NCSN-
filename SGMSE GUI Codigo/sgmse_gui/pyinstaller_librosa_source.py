import importlib.util
import sys
from pathlib import Path


LIBROSA_ROOT = Path(sys._MEIPASS) / "librosa"


class LibrosaSourceFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "librosa" and not fullname.startswith("librosa."):
            return None

        module_path = LIBROSA_ROOT.joinpath(*fullname.split(".")[1:])
        package_init = module_path / "__init__.py"
        if package_init.is_file():
            return importlib.util.spec_from_file_location(
                fullname,
                package_init,
                submodule_search_locations=[str(module_path)],
            )

        module_file = module_path.with_suffix(".py")
        if module_file.is_file():
            return importlib.util.spec_from_file_location(fullname, module_file)
        return None


sys.meta_path.insert(0, LibrosaSourceFinder())
