import os
import os.path as osp
import logging
import time
import matplotlib.pyplot as plt
import pickle

# call example: from file_management import SaveFig, SaveData, LoadData, MakeLogger, TimeIt


def MakeDirIfNotDir(path):
    """Create a directory if it does not exist."""
    if not osp.isdir(path):
        os.makedirs(path)
    return path

"""Create result directories in the current working directory."""
RESULT_DIR = osp.join(osp.dirname(osp.abspath(__file__)), 'results')
TIMERS_DIR = osp.join(RESULT_DIR, 'timers')
FIGURES_DIR = osp.join(RESULT_DIR, 'figures')
DATA_DIR = osp.join(RESULT_DIR, 'data')  # saved data *not the input dataset*
for dir_path in [RESULT_DIR, TIMERS_DIR, FIGURES_DIR, DATA_DIR]:
    MakeDirIfNotDir(dir_path)


def MakeLogger(debugMode=False):



    # Create and config logger
    global RESULT_DIR
    logger_path = osp.join(RESULT_DIR, 'logger.log')
    if osp.isfile(logger_path):
        # delete the file if it exists
        os.remove(logger_path)

    debug_level = logging.DEBUG if debugMode else logging.INFO
    LOG_FORMAT = "%(levelname)s: %(asctime)s  %(message)s"
    logging.basicConfig(filename=osp.join(RESULT_DIR, 'logger.log'),
                        level=debug_level,  # can be DEBUG, INFO, WARNING, ERROR, CRITICAL
                        format=LOG_FORMAT)
    logger = logging.getLogger()

    logger.info(f"start logger")
    return logger

logger = MakeLogger()

def TimeIt(func):
    """Decorator that write functions' execution time to text file in TIMERS_DIR/function name.txt"""
    global TIMERS_DIR
    firstTime = True
    file = open(osp.join(TIMERS_DIR, func.__name__ + ".txt"), mode="a")
    def wrapper(*args, **kwargs):
        nonlocal firstTime, file
        # Check if the first argument is a class instance, if it does change the file name at the beginning
        if firstTime:
             if len(args) > 0  and hasattr(args[0], "__class__"):
                file.close()
                file = open(osp.join(TIMERS_DIR, f"{args[0].__class__.__name__}_{func.__name__}.txt"), mode="a")
                firstTime = False
        ts = time.time()
        result = func(*args, **kwargs)
        file.write(f'{time.time() - ts}\n')
        return result
    return wrapper


def SaveFig(figName):
    """    Save the figure to the figures directory with the given name.    """
    global FIGURES_DIR
    if ".png" not in figName and ".jpg" not in figName:
        figName = figName + ".png"
    plt.savefig(osp.join(FIGURES_DIR, figName), bbox_inches='tight', dpi=300)


def SaveData(data, dataName):
    """    Save the data to a pickle file in the data directory with the given name.     """

    global DATA_DIR
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError("DATA_DIR is not a valid directory: {}".format(DATA_DIR))
    dataPath = osp.join(DATA_DIR, dataName + ".pkl")

    if osp.isfile(dataPath):
        # delete the file
        os.remove(dataPath)

    with open(dataPath, 'wb') as outf:
        pickle.dump(data, outf, pickle.HIGHEST_PROTOCOL)

    return dataPath

def LoadData(dataName):
    """
    Load the data from a pickle file in the data directory with the given name.
    """
    global DATA_DIR
    dataPath = osp.join(DATA_DIR, dataName + ".pkl")
    if not osp.isfile(dataPath):
        raise RuntimeError("data file does not exist: {}".format(dataPath))
    with open(dataPath, 'rb') as inf:
        data = pickle.load(inf)
    return data