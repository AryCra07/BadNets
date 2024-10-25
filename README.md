# Badnets

A simple implementation of [*Badnets: Identifying vulnerabilities in the machine learning model supply
chain* ](https://arxiv.org/abs/1708.06733).

The CNN I built consists of three convolutional layers and two fully connected layers. Taking into account the need to
support both MNIST and CIFAR10 datasets, this model incorporates an extra convolutional
layer when compared to the model presented in the original paper.

The implementation strategy is as follows:

- First, train the model using a partially randomly poisoned training set.
- Then, test the model using a clean test set and a test set fully containing triggers, obtaining metrics such as BA(
  Benign accuracy) and
  ASR(Attack Success Rate).

## INSTALLATION

To install the required packages, you can run the following command:

```shell
pip install -r requirements.txt
```

## USAGE

To run the code, you can use the following command:

```shell
python main.py
```

You can customize various parameters on the command line:

```shell
python main.py --help
```

## LICENSE

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.