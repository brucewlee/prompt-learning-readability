def predefined_args(parser):
    parser.add_argument(
        '--epochs',
        default=30,
        type=int,
        help='(int) Number of training epochs'
        )
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int,
        help='Number of batches - depending on the max sequence length and GPU memory. For 512 sequence length batch of 10 works without cuda memory issues. For small sequence length can try batch of 32 or higher.'
        )
    parser.add_argument(
        '--max_seq_length',
        default=512,
        type=int,
        help='(int) Pad or truncate text sequences to a specific length. If `None` it will use maximum sequence of word piece tokens allowed by model.' 
        )
    parser.add_argument(
        '--model_name_or_path',
        default="facebook/bart-base",
        type=str,
        help='(str) Name of transformers model - will use already pretrained model. Path of transformer model - will load your own model from local disk.'
        )
    parser.add_argument(
        '--dataset',
        default='onestopenglish',
        type=str,
        help='(str) Train tsv/csv path.'
        )
    parser.add_argument(
        '--tokenizer',
        default="facebook/bart-base",
        type=str,
        help='Name of tokenizer'
        )
    parser.add_argument(
        '--learning_rate',
        default=1e-5,
        type=float,
        help='learning rate'
        )
    parser.add_argument(
        '--config',
        default=None,
        type=str,
        help='Name of config'
        )
    parser.add_argument(
        '--save',
        default=True,
        type=bool,
        help='(bool) to save or not'
        )
    parser.add_argument(
        '--earlystop',
        default=True,
        type=bool,
        help='(bool) to early stop or not'
        )
    return parser