from spirl.utils.general_utils import AttrDict
from spirl.data.umaze.src.umaze_data_loader import UmazeSequenceSplitDataset


data_spec = AttrDict(
    dataset_class=UmazeSequenceSplitDataset,
    n_actions=2,
    state_dim=2,
    env_name="roommaze-v1",
    res=128,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 500
