class Error(Exception):
    pass


class TSPError(Error):
    def __init__(self, message):
        Exception.__init__(self, message)