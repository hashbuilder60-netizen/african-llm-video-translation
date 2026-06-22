from opustools import OpusRead

print("Downloading Ibibio-English corpus from JW300...")
print("This may take a few minutes...")

opus_reader = OpusRead(
    directory="JW300",
    source="en",
    target="ibb",
    write=["jw300.en", "jw300.ibb"],
    write_mode="moses",
    download_dir="corpus"
)

opus_reader.printPairs()

print("Download complete!")
print("Check for jw300.en and jw300.ibb files")