import os

main_folder_path = "/Users/cbainton/Downloads/st_genecounts_outs"
out_dir = "/Users/cbainton/Downloads/out_st_full_structure"
zip_files = os.listdir(main_folder_path)
zip_files.remove(".DS_Store")

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for zip_file in zip_files:
    print("Zip: " +  zip_file)
    if os.path.exists(out_dir + "/outs"):
        print ("Careful, will overwrite outs")

    bash_zip = zip_file.replace('(', '\\(').replace(')', '\\)')
    os.system("unzip " + main_folder_path + "/" + bash_zip + " " + "-d " + out_dir)
    outs_folder = out_dir + "/" + "outs"
    summary_file = open(outs_folder + "/" + "web_summary.html", "r")
    sum_text = summary_file.read()
    start = sum_text.find("\"Sample ID\", \"")
    sample_name = sum_text[start + 14 : start+ 26] # Get sample name from end of ' "Sample ID, "" '
    sample_folder = out_dir + "/" + sample_name
    print(sample_folder)

    if os.path.exists(sample_folder):
        print ("Overwriting sample folder " + sample_folder)
        if os.path.exists(sample_folder + "/" + "outs"):
            i = 1
            while os.path.exists(sample_folder + "/" + "outs_" + i.__str__()):
                i += 1
            os.rename(outs_folder , sample_folder + "/" + "outs_" + i.__str__())
    else:
        os.mkdir(sample_folder)
        os.rename(outs_folder , sample_folder + "/" + "outs")