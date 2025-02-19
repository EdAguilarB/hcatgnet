from enum import Enum


class ProblemTypes(str, Enum):
    Classification = "classification"
    Regression = "regression"


class Networks(str, Enum):
    GCN = "gcn"
    GAT = "gat"
    GRAPHSAGE = "graphsage"


class Pooling(str, Enum):
    GlobalMaxPool = "gmp"
    GlobalAddPool = "gsp"
    GlobalMeanPool = "gmeanp"
    GlobalMeanMaxPool = "gmeanmaxp"


class Optimizers(str, Enum):
    Adam = "adam"
    SGD = "sgd"
    RMSprop = "rmsprop"


class Schedulers(str, Enum):
    StepLR = "steplr"
    MultiStepLR = "multisteplr"
    ExponentialLR = "exponentiallr"
    ReduceLROnPlateau = "reducelronplateau"
