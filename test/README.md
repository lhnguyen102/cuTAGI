# Unit Tests Usage Instructions

The program can be used with the following command-line options:

- Perform Tests: `build/main test`
- Run one specific test: `build/main test [architecture-name]`
- Reinizialize all test outputs: `build/main test -reset all`
- Reinizialize one specific output: `build/main test -reset [architecture-name]`
- Available architectures: *fnn, fnn_heteros, fnn_full_cov, fnn_derivates, cnn, cnn_batch_norm, autoencoder, lstm, cnn_resnet*
