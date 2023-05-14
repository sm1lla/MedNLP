import typer

cli = typer.Typer(name="TODO: find name")


@cli.command()
def moin():
    print("moin")


@cli.command()
def test():
    from .train import train

    train()


@cli.command()
def main(output_file_name):
    pass


@cli.command()
def dataset():
    from .dataset import examine_dataset

    examine_dataset()


@cli.command()
def train():
    from .train import train

    train()


@cli.command()
def infer(text: str, path: str):
    from .inference import infer

    infer(text, path)
