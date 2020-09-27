import torch


"""
Parameters

    d_model – the number of expected features in the encoder/decoder inputs (default=512).

    nhead – the number of heads in the multiheadattention models (default=8).

    num_encoder_layers – the number of sub-encoder-layers in the encoder (default=6).

    num_decoder_layers – the number of sub-decoder-layers in the decoder (default=6).

    dim_feedforward – the dimension of the feedforward network model (default=2048).

    dropout – the dropout value (default=0.1).

    activation – the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).

    custom_encoder – custom encoder (default=None).

    custom_decoder – custom decoder (default=None).
"""
trfmr_config = {
    'd_model': 256,  # number of features in embedding
    'nhead': 8,  # number of attention heads
    'num_encoder_layers': 8,
    'num_decoder_layers': 8,
    'dim_feedforward': 2048,
    'activation': 'relu',
}

opt_config = {
    'lr': 3e-4,
    'beta1': 0.5,
    'beta2': 0.999,
    'num_epochs': 300
}


def main():
    # Initialize model.
    trfm_model = torch.nn.Transformer(**trmfr_config)

    # Initialize optimizer.
    opt = torch.optim.AdamW(trfm.parameters(), opt_config['lr'],
                            [opt_config['beta1'], opt_config['beta2']])

    # Set loss function.
    loss_fn = torch.nn.BCELoss

    # Load Data.
    # TODO: implement data loading.
    data = []

    # Training loop
    for epoch in range in range(opt_config['num_epochs']):
        for expected_out, batch in data:
            opt.zero_grad()
            actual_out = trfm_model(batch)
            loss = torch.nn.BCELoss(actual_out, expected_out)
            loss.backward()
            opt.step()


if __name__ == '__main__':
    main()