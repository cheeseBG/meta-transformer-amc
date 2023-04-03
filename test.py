import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tnrange

from util import extract_test_sample

def test(model, test_x, test_y, n_way, n_support, n_query, test_episode):
    """
    Tests the protonet
    Args:
        model: trained model
        test_x (np.array): dataloader dataframes of testing set
        test_y (np.array): labels of testing set
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
        test_episode (int): number of episodes to test on
    """
    conf_mat = torch.zeros(n_way, n_way)
    running_loss = 0.0
    running_acc = 0.0

    '''
    Modified
    # Extract sample just once
    '''
    sample = extract_test_sample(n_way, n_support, n_query, test_x, test_y)
    query_samples = sample['q_csi_mats']

    # Create target domain Prototype Network with support set(target domain)
    z_proto = model.create_protoNet(sample)

    total_count = 0
    for episode in tnrange(test_episode, desc="test"):
        for label, q_samples in enumerate(query_samples):
            for i in range(0, len(q_samples)//n_way):
                output = model.proto_test(q_samples[i*n_way:(i+1)*n_way], z_proto, n_way, label)
                a = output['y_hat'].cpu().int()
                for cls in range(n_way):
                    conf_mat[cls, :] = conf_mat[cls, :] + torch.bincount(a[cls, :], minlength=n_way)

                running_acc += output['acc']
                total_count += 1

    avg_acc = running_acc / total_count
    print('Test results -- Acc: {:.4f}'.format(avg_acc))
    return (conf_mat / (test_episode * n_query), avg_acc)

