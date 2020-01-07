class NaiveBayes(object):
    def __init__(self, dataset, target_column):
        self.dataset = dataset
        self.target = target_column
    def column_classes(self, column):
        return list(set(self.dataset[column]))
    def count_instance(self, column):
        return {k: self.dataset[column].count(k) for k in self.column_classes(column)}
    def sample_size(self):
        return len(self.dataset[self.target])
    def response_frequency(self, column):
        responses = self.column_classes(self.target)
        frequency = {k: {v: 0 for v in responses} for k in self.column_classes(column)}
        for value, response in zip(self.dataset[column], self.dataset[self.target]):
            frequency[value][response] += 1
        return frequency
    def likelihood(self, column, evidence, response):
        frequency = self.response_frequency(column)
        response_count = self.count_instance(self.target)
        prior_probablity = response_count[response] / self.sample_size()
        conditional_probablity = frequency[evidence][response] / response_count[response]
        p_evidence = self.count_instance(column)[evidence] / self.sample_size()
        return (conditional_probablity * prior_probablity) / p_evidence
    def esmitate(self, response, **evidences):
        # evidences must come in this format {'column': 'evidence'}
        result = 1
        evidences = evidences['evidences']
        for column in evidences:
            result *= self.likelihood(column, evidences[column], response)
        return result
    def row_generator(self, dataset):
        row = {k: None for k in dataset.keys()}
        row.pop(self.target)
        for i in range(len(dataset[self.target])):
            for key in row:
                row[key] = dataset[key][i]
            yield (row, i)
    def evaluate(self, test_dataset):
        tp, fp, tn, fn = 0, 0, 0, 0
        for row, i in self.row_generator(test_dataset):
            actual_response = test_dataset[self.target][i]
            positive = 'recurrence-events'
            negative = 'no-recurrence-events'
            correct_guesses = 0
            if actual_response == positive:
                if self.esmitate(positive, evidences=row) > self.esmitate(negative, evidences=row):
                    tp += 1
                elif self.esmitate(negative, evidences=row) > self.esmitate(positive, evidences=row):
                    fn += 1
            elif actual_response == negative:
                if self.esmitate(positive, evidences=row) > self.esmitate(negative, evidences=row):
                    fp += 1
                elif self.esmitate(negative, evidences=row) > self.esmitate(positive, evidences=row):
                    tn += 1
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        result = {'recall': recall, 'accuracy': accuracy, 'precision': precision}
        return result
            
