
with open("public/v4dv_bg.wasm", "rb") as f:
    wasm_code = f.read()
html_code = f"""
<!doctype html>
<html>

<head>
    <meta charset="utf-8">
    <title>v4dv</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            background-color: black;
            color: white;
            font-family: Arial, Helvetica, sans-serif;
        }}
    </style>

</head>

<body>
    <script src="./v4dv.js"></script>
    <script>
        const {{run_wasm,initSync}} = wasm_bindgen;
        const wasm_code = new Uint8Array([{",".join([str(x) for x in wasm_code])}])
        initSync(wasm_code);
        run_wasm();
    </script>
</body>

</html>
"""

with open("public/viewer_packed.html", "w") as f:
    f.write(html_code)