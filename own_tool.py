import sys
from pylint.lint import Run as pylinter
from pep8 import Checker, BaseReport
from vulture import Vulture
import traceback

from isort import SortImports

file_name = sys.argv[1]


class MyLinters(object):
    def __init__(self, passfile):
        self.passfile = passfile

    def my_pylinter(self):
        """
        Pylint is a source-code, bug and quality checker for the
        Python programming language. It follows the style recommended by PEP 8,
        the Python style guide.
        """
        print('Running pylinter...')
        results = pylinter([self.passfile])
        # print(results.linter.stats(['']))
        print(results.linter.stats['global_note'])

    def my_pep8(self):
        """
        pycodestyle is a tool to check your Python code against some of the style conventions in PEP 8.
        """
        print('Running pep8...')
        check_file = Checker(self.passfile).check_all()
        reports = BaseReport(check_file)
        print(reports)
        a = reports.print_statistics()
        print(a)

    def dead_code(self):
        """
         Function used to identify dead/unused code
        """
        print('checking for dead code...')
        file_names = []
        file_names.append(file_name)
        vulture = Vulture()
        vulture.scavenge(file_names)
        for item in vulture.get_unused_code():
            print(item.filename, item.message, item.first_lineno, item.last_lineno, item.confidence)

    def last_runtime_error(self):
        """
        Can be used to invoke a runtime error without directly running the script
        """
        print('checking for runtime errors...')
        try:
            import test_file2 # needs to be imported, cannot use argv
        except Exception as e:
            (exc_type, exc_value, exc_traceback) = sys.exc_info()
            trace_back = [traceback.extract_tb(sys.exc_info()[2])[-1]][0]
            print("Exception {} is on line {}".format(exc_type, trace_back[1]))

    def sort_imports(self):
        """
        Function used to sort imports
        """
        print('Sorting imports...')
        SortImports(self.passfile)


# obj = MyLinters(file_name)
# obj.sort_imports()

def run_linters():
    obj = MyLinters(file_name)
    for i in dir(obj):
        item = getattr(obj, i)
        if callable(item) and i.startswith('m'):
            item()


run_linters()
