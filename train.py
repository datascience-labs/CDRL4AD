import time
import json
from datetime import datetime
import torch.nn as nn

from args import get_parser
from utils import *
from cdrl4ad import CDRL4AD
from prediction import Predictor
from training import Trainer

if __name__ == "__main__":
    process_start_time = time.time()

    parser = get_parser()
    args = parser.parse_args()

    dataset = args.dataset
    window_size = args.window_size
    causal_window_size = args.causal_window_size
    causal_hid_dim = args.causal_hid_dim
    topk = args.topk
    batch_size = args.batch_size

    device = args.device
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    
    spec_res = args.spec_res
    normalize = args.normalize
    n_epochs = args.epochs
    embed_dim = args.embed_dim
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    causal_thres = args.causal_thres
    group_index = args.group[0]
    index = args.group[2:]
    output_path = f'{args.save_path}/{dataset}'
    args_summary = str(args.__dict__)
    print(args_summary)

    if dataset == 'SMD':
        id = datetime.now().strftime("%d%m%Y_%H%M%S") + "_" + args.group
    else:
        id = datetime.now().strftime("%d%m%Y_%H%M%S")

    if dataset == 'SMD':
        (x_train, _), (x_test, y_test), root_cause_labels = get_data(f"/datasets/{dataset}", dataset, batch_size, normalize=normalize, group=args.group)
    elif dataset in ['SWaT', 'PSM', 'HAI', 'WADI', 'creditcard', 'GECCO', 'JumpStarter', 'SMAP', 'MSL']:
        (x_train, _), (x_test, y_test), _ = get_data(f"/datasets/{dataset}", dataset, batch_size, normalize=normalize)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    save_path = f"{output_path}/{id}"

    x_train = torch.from_numpy(np.array(x_train)).float()
    x_test = torch.from_numpy(np.array(x_test)).float()
    n_features = x_train.shape[1]


    target_dims = None
    
    if target_dims is None:
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size)
    test_dataset = SlidingWindowDataset(x_test, window_size)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, test_dataset=test_dataset
    )

    model = CDRL4AD(
        n_features,
        topk,
        window_size,
        causal_window_size,
        causal_hid_dim,
        out_dim,
        device,
        embed_dim,
        causal_thres,
        gru_n_layers=1,
        forecast_n_layers=3,
        recon_n_layers=1,
        dropout=0.3,
        alpha=0.2
    )

    print(f"Model Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        device,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
    )
    
    trainer.fit(train_loader, val_loader)

    plot_losses(trainer.losses, save_path=save_path, plot=False)

    test_loss = trainer.evaluate(test_loader)
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test reconstruction loss: {test_loss[1]:.5f}")
    print(f"Test total loss: {test_loss[2]:.5f}")

    
    trainer.load(f"{save_path}/model.pt")
    prediction_args = {
        'topk':topk,
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': False,
        "level": 0.90,
        "q": 0.001,
        'dynamic_pot': False,
        "use_mov_av": False,
        "gamma": 0.3,
        "reg_level": 0,
        "save_path": save_path,
    }
    best_model = trainer.model

    predictor = Predictor(
        device,
        best_model,
        window_size,
        n_features,
        batch_size,
        prediction_args,
    )

    file_path = os.path.join(save_path, "arguments.txt")
    with open(file_path, 'w') as file:
        file.write(args_summary)

    label = y_test[window_size:] if y_test is not None else None
    prediction_start_time = time.time()
    if dataset == 'SMD':
        root_cause_labels = root_cause_labels[window_size:]
        predictor.predict_anomalies(x_train, x_test, true_anomalies=label, root_cause_labels=root_cause_labels)
    else:
        predictor.predict_anomalies(x_train, x_test, true_anomalies=label)
    prediction_end_time = time.time()
    prediction_duration = prediction_end_time - prediction_start_time
