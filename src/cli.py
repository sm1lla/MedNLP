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
    