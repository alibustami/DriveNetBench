"""Command Line Interface for DriveNetBench."""

from typer import Typer

app = Typer()


@app.command()
def run():
    """Run DriveNetBench."""
    print("Running DriveNetBench.")


@app.command()
def test():
    """Test DriveNetBench."""
    print("Testing DriveNetBench.")


if __name__ == "__main__":
    app()
