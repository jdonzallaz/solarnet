import typer

# Init CLI
app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})


@app.command()
def train():
    typer.echo(f"Not implemented")


@app.command()
def test():
    typer.echo(f"Not implemented")


if __name__ == "__main__":
    app()
