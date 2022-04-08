from spirl.utils.general_utils import AttrDict
from spirl.data.sawyer.src.sawyer_data_loader import SawyerSequenceSplitDataset


data_spec = AttrDict(
    dataset_class=SawyerSequenceSplitDataset,
    n_actions=4,
    state_dim=39,
    env_name="reach-v0",
    res=128,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 500
