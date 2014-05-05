import os


def results_dir(filename = None):
    """Return the directory you should write results to. Creates the directory if it
    doesn't exist. Raises Exception if the directory cannot be created, is not
    a directory or is not writeable."""
    path = 'results'
    if os.path.isdir(path):
        if not os.access(path, os.R_OK | os.W_OK):
            raise EnvironmentError("{0} is not readable or writable".format(os.path.abspath(path)))
        return os.path.join(path, filename) if filename else path
    os.mkdir(path)  # raises if it fails
    return os.path.join(path, filename) if filename else path
