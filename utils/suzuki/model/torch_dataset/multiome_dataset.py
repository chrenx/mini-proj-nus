import numpy as np
import scipy.sparse
import torch

from utils.suzuki.utility.get_selector_with_metadata_pattern import get_selector_with_metadata_pattern
from utils.suzuki.utility.metadata_utility import CELL_TYPES, MALE_DONOR_IDS

METADATA_KEYS = [
    "day",
    "nonzero_ratio",
    "nonzero_q25",
    "nonzero_q50",
    "nonzero_q75",
    "mean",
    "std",
]


class MultiomeDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""

    def __init__(
        self, inputs_values, preprocessed_inputs_values, metadata, 
              targets_values, preprocessed_targets_values, selected_metadata, opt=None
    ):
        self.opt = opt
        # selected_metadata 是none
        # assert isinstance(inputs_values, scipy.sparse.csr_matrix)
        if selected_metadata is not None:
            selector = get_selector_with_metadata_pattern(metadata=metadata, metadata_pattern=selected_metadata)
            inputs_values = inputs_values[selector, :]
            preprocessed_inputs_values = preprocessed_inputs_values[selector, :]
            metadata = metadata.loc[selector, :]
            if targets_values is not None:
                targets_values = targets_values[selector, :]
                preprocessed_targets_values = preprocessed_targets_values[selector, :]
        if targets_values is not None:
            # assert isinstance(targets_values, scipy.sparse.csr_matrix)
            assert preprocessed_inputs_values.shape[0] == targets_values.shape[0]
            assert preprocessed_targets_values.shape[0] == targets_values.shape[0]
        self.preprocessed_inputs_values = preprocessed_inputs_values
        self.targets_values = targets_values
        self.preprocessed_targets_values = preprocessed_targets_values

        assert preprocessed_inputs_values.shape[0] == len(metadata)
        self.metadata = metadata
        gender_ids = np.zeros((len(self.metadata),), dtype=int)
        gender_ids[self.metadata["donor"].isin(MALE_DONOR_IDS)] = 1

        self.gender_ids = gender_ids # (105942,)
        self.day_ids = self.metadata["day"].to_numpy() # (105942,)
        mapping = {13176: 1, 31800: 2, 32606: 3}
        self.donor_ids = self.metadata['donor'].map(mapping).to_numpy() # (105942,)

        # print("day: ", self.day_ids.shape)
        # print(self.day_ids[:4])
        # print("gender_ids: ", self.gender_ids.shape)
        # print(self.gender_ids[:4])
        # print("donor_ids: ", self.donor_ids.shape)
        # print(self.donor_ids[:4])
        # exit(0)

        cell_type_ids = np.zeros((len(self.metadata),), dtype=int)
        cell_type_values = self.metadata["cell_type"].values
        for i, cell_type in enumerate(CELL_TYPES):
            cell_type_ids[cell_type_values == cell_type] = i
        self.cell_type_ids = cell_type_ids
        # self.cell_type_one_hot = np.eye(len(CELL_TYPES))[self.cell_type_ids]
        self.metadata_keys = METADATA_KEYS
        # print(metadata.columns)

    def __getitem__(self, index):
        if isinstance(self.preprocessed_inputs_values, scipy.sparse.csr_matrix):
            preprocessed_inputs_values = self.preprocessed_inputs_values[index].toarray().ravel()
        else:
            preprocessed_inputs_values = self.preprocessed_inputs_values[index].ravel()
        preprocessed_inputs_tensor = torch.as_tensor(preprocessed_inputs_values, dtype=torch.float32)

        gender_id = torch.as_tensor(self.gender_ids[index], dtype=torch.int64)
        info = torch.as_tensor(self.metadata.iloc[index, :][self.metadata_keys].values.astype(float), 
                               dtype=torch.float32)
        day_id = torch.as_tensor(self.day_ids[index], dtype=torch.int64)
        donor_id = torch.as_tensor(self.donor_ids[index], dtype=torch.int64)
        
        if self.targets_values is not None:
            if isinstance(self.targets_values, scipy.sparse.csr_matrix):
                targets_values = self.targets_values[index].toarray().ravel()
            else:
                targets_values = self.targets_values[index].ravel()
            preprocessed_targets_values = self.preprocessed_targets_values[index].ravel()
            targets_tensor = torch.as_tensor(targets_values, dtype=torch.float32)
            preprocessed_targets_tensor = torch.as_tensor(preprocessed_targets_values, dtype=torch.float32)
            # print("preprocessed_inputs_tensor :", preprocessed_inputs_tensor.shape)  # 256
            # print("gender_id                  :", gender_id)                         # 1或0
            # print("info                       :", info.shape)                        # 7
            #       # day_id                                                           # 2 || 3 || 4
            #       # donor_id          mapping = {13176: 1, 31800: 2, 32606: 3}       # 1 || 2 || 3                                      
            # print("targets_tensor             :", targets_tensor.shape)              # 23418
            # print("preprocessed_targets_tensor:", preprocessed_targets_tensor.shape) # 128

            return [preprocessed_inputs_tensor, gender_id, info, day_id, donor_id,
                    targets_tensor, preprocessed_targets_tensor]
            # if self.opt.backbone == "unet":
            #     return [preprocessed_inputs_tensor, gender_id, info, day_id, donor_id,
            #             targets_tensor, preprocessed_targets_tensor]
            # elif self.opt.backbone == "mlp":
            #     return [preprocessed_inputs_tensor, gender_id, info, 
            #             targets_tensor, preprocessed_targets_tensor]
        else:
            print("不对...")
            raise RuntimeError
            return [
                preprocessed_inputs_tensor,
                gender_id,
                info,
            ]

    def __len__(self):
        return self.preprocessed_inputs_values.shape[0]
