import numpy as np
class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            indices = np.random.choice(data_length, data_length, replace=True)
            self.indices_list.append(indices)

    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag_indices in self.indices_list:
            model = model_constructor()
            data_bag, target_bag = data[bag_indices], target[bag_indices]
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target

    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        predictions = np.zeros((len(data), self.num_bags))
        for i, model in enumerate(self.models_list):
            predictions[:, i] = model.predict(data)
        return np.mean(predictions, axis=1)

    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during the training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        for i, model in enumerate(self.models_list):
            oob_indices = list(set(range(len(self.data))) - set(self.indices_list[i]))
            oob_data = self.data[oob_indices]
            predictions = model.predict(oob_data)
            for j, idx in enumerate(oob_indices):
                list_of_predictions_lists[idx].append(predictions[j])
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)

    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from the training set.
        If an object has been used in all bags during the training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = np.array([np.mean(pred) if len(pred) > 0 else None for pred in self.list_of_predictions_lists])

    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        valid_indices = [i for i in range(len(self.oob_predictions)) if self.oob_predictions[i] is not None]
        mse = np.mean((self.target[valid_indices] -  self.oob_predictions[valid_indices]) ** 2)
        return mse