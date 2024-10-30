import torch
from sklearn.decomposition import TruncatedSVD

_default_svd_params = {
    "n_components": 128,
    "random_state": 42,
    "n_oversamples": 20,
    "n_iter": 7,
}

# from cuml.decomposition import TruncatedSVD

class IterativeSVDImputator(object):
    def __init__(self, svd_params=_default_svd_params, iters=2, device='cpu'):
        self.missing_values = 0.0  # Define missing values as zero
        self.svd_params = svd_params
        self.iters = iters
        self.svd_decomposers = [None for _ in range(self.iters)]
        self.device = device

    def fit(self, X):
        if isinstance(X, torch.Tensor):
            # Move data to GPU if not already a tensor
            # print("tSVD fit: ", X.shape)
            X = torch.tensor(X).to(self.device) if not isinstance(X, torch.Tensor) else X
            mask = X == self.missing_values
            transformed_X = X.clone()

            for i in range(self.iters):
                # Convert to numpy for cuML compatibility and perform SVD
                svd = TruncatedSVD(**self.svd_params)
                self.svd_decomposers[i] = svd

                # Transfer data back to CPU, perform SVD fit, and return to GPU
                X_cpu = transformed_X.cpu().numpy()
                svd.fit(X_cpu)
                new_X = torch.tensor(svd.inverse_transform(svd.transform(X_cpu))).to(self.device)
                transformed_X[mask] = new_X[mask]
        else:
            mask = X == self.missing_values
            transformed_X = X.copy()
            for i in range(self.iters):
                self.svd_decomposers[i] = TruncatedSVD(**self.svd_params)
                self.svd_decomposers[i].fit(transformed_X)
                new_X = self.svd_decomposers[i].inverse_transform(self.svd_decomposers[i].transform(transformed_X))
                transformed_X[mask] = new_X[mask]

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            # print("tSVD transform: ", X.shape)
            # Move data to GPU if not already a tensor
            X = torch.tensor(X).to(self.device) if not isinstance(X, torch.Tensor) else X
            mask = X == self.missing_values
            transformed_X = X.clone()

            for i in range(self.iters):
                # Apply each SVD decomposer to transform and inverse transform the data
                X_cpu = transformed_X.cpu().numpy()  # Transfer to CPU for tSVD
                svd = self.svd_decomposers[i]
                new_X = torch.tensor(svd.inverse_transform(svd.transform(X_cpu))).to(self.device)            
                transformed_X[mask] = new_X[mask]
            return transformed_X
        else:
            mask = X == self.missing_values
            transformed_X = X.copy()
            for i in range(self.iters):
                new_X = self.svd_decomposers[i].inverse_transform(self.svd_decomposers[i].transform(transformed_X))
                transformed_X[mask] = new_X[mask]
            return transformed_X


# class IterativeSVDImputator(object):
#     def __init__(self, svd_params=_default_svd_params, iters=2):
#         self.missing_values = 0.0
#         self.svd_params = svd_params
#         self.iters = iters
#         self.svd_decomposers = [None for _ in range(self.iters)]

#     def fit(self, X):
#         mask = X == self.missing_values
#         transformed_X = X.copy()
#         for i in range(self.iters):
#             self.svd_decomposers[i] = TruncatedSVD(**self.svd_params)
#             self.svd_decomposers[i].fit(transformed_X)
#             new_X = self.svd_decomposers[i].inverse_transform(self.svd_decomposers[i].transform(transformed_X))
#             transformed_X[mask] = new_X[mask]

#     def transform(self, X):
#         mask = X == self.missing_values
#         transformed_X = X.copy()
#         for i in range(self.iters):
#             new_X = self.svd_decomposers[i].inverse_transform(self.svd_decomposers[i].transform(transformed_X))
#             transformed_X[mask] = new_X[mask]
#         return transformed_X
