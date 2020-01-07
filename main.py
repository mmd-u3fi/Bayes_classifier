from dataset_parser import parse_dataset
from bayes import NaiveBayes
from test_train_split import dataset_split

dataset = parse_dataset()
(train , test) = dataset_split(dataset, 0.2)
naive_bayes = NaiveBayes(train, 'class')
score = naive_bayes.evaluate(test)

print(score)