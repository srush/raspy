jupyter nbconvert --to markdown --execute --template template/ Blog.ipynb
pandoc docs/header-includes.yaml Blog.md --output=docs/index.html --to=html5 --css=docs/github.min.css --css=docs/tufte.css --no-highlight --self-contained --metadata pagetitle="Thinking like Transformer"
