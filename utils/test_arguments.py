def predefined_args(parser):
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
        default="ckpt/facebook-bart-base-onestopenglish-RF-1e-05-30epochs",
        type=str,
        help='(str) Name of transformers model - will use already pretrained model. Path of transformer model - will load your own model from local disk.'
        )
    parser.add_argument(
        '--dataset',
        default='commoncore',
        type=str,
        help='(str) onestopenglish or newsela or cambridge or commoncore'
        )
    parser.add_argument(
        '--tokenizer',
        default="facebook/bart-base",
        type=str,
        help='Name of tokenizer'
        )
    parser.add_argument(
        '--config',
        default=None,
        type=str,
        help='Name of config'
        )
    return parser #terminal 1

"""def predefined_args(parser):
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
        default="ckpt/facebook-bart-base-onestopenglish-RF-1e-05-30epochs-newsela-RF-1e-05-3epochs",
        type=str,
        help='(str) Name of transformers model - will use already pretrained model. Path of transformer model - will load your own model from local disk.'
        )
    parser.add_argument(
        '--dataset',
        default='cambridge',
        type=str,
        help='(str) onestopenglish or newsela or cambridge or commoncore'
        )
    parser.add_argument(
        '--tokenizer',
        default="facebook/bart-base",
        type=str,
        help='Name of tokenizer'
        )
    parser.add_argument(
        '--config',
        default=None,
        type=str,
        help='Name of config'
        )
    return parser"""