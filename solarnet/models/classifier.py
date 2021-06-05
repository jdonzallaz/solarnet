from torch import nn


class Classifier(nn.Module):
    """
    A simple classifier model. Mostly used as a head on more complex models.
    Return a linear head if n_hidden is None.
    No softmax is applied. Use cross-entropy as training loss.

    :param n_input: size of the input
    :param n_class: number of class as output
    :param n_hidden: number of neurons in the hidden layer.
    :param dropout: dropout rate to apply.
    """

    def __init__(self, n_input, n_class, n_hidden=512, dropout=0.2):
        super().__init__()

        if n_hidden is None:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout),
                nn.Linear(n_input, n_class),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout),
                nn.Linear(n_input, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(n_hidden, n_class),
            )

    def forward(self, x):
        return self.classifier(x)
