import nox

# Settings
nox.options.reuse_existing_virtualenvs = True
PYTHON_VERSIONS = ["3.10.14"]
LOCATIONS = ["src", "tests", "noxfile.py"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    session.install("pytest")
    session.install("datasets", "transformers")
    session.install("-e", ".", "--no-deps")
    session.run("pytest", "--maxfail=1", "--disable-warnings", "-q")


@nox.session(python=PYTHON_VERSIONS)
def lint(session):
    session.install("ruff")
    session.run("ruff", "check", "--fix", *LOCATIONS)


@nox.session(python=PYTHON_VERSIONS)
def format(session):
    session.install("ruff")
    session.run("ruff", "format", *LOCATIONS)
