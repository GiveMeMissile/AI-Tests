This test compares the usage of different datasets used for similar pieces of code that trains an AI.
The first test utilizes the datasets in this respositories using Placeholder for testing and Bad_data for training.
This compares fish to non fish images.
As expected the first test didn't preform so well with a highest test accuracy of 62.4% in 100 epoches with both the testing and training data being subpar.
The second test utilizes a better Dataset being a MNIST dataset from Kaggle as found here https://www.kaggle.com/datasets/rakuraku678/mnist-60000-hand-written-number-images/data.
This compares hand written numbers.
This dataset prefromed much better with a highest test accuracy of 90% in just 20 epoches.
Which is much better than the pitiful 62.4% seen in the first test that required 100 epoches to achive that result.
So basically don't use subpar datasets.
