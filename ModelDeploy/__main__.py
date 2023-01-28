# entry point
from . import main

args = main.parse_args()
main.main(args=args)
