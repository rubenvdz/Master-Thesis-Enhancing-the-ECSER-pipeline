import bibtexparser
#import csv
import pandas as pd
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt

# Load entries from the .bib files
entries = []
for file in ["./Bib/acm_ICSE.bib", "./Bib/acm_FSE.bib", "./Bib/acm_TSE.bib"]:
    library = bibtexparser.parse_file(file)
    entries.extend(library.entries)


def duplicates(entries):
    # Check for duplicate titles 
    title_set = set()
    duplicates = []
    
    for entry in entries:
        title = entry['title']
        if title:
            if title in title_set:
                duplicates.append(title)
            else:
                title_set.add(title)
    
    if duplicates:
        print("\nDuplicate titles found:")
        for dup in duplicates:
            print(dup)
    else:
        print("\nNo duplicate titles found.")


def keyword_search(entries,keywords):
    # Search the title and abstract for keywords
    count = 0
    keyword_counter = Counter() # Count which keywords are matched the most (for checking if any are redundant)
    matches = [] # Create a list of true or false, for whether the keyword appears
    for entry in entries:
        found = False
        for kw in keywords:
            if (kw in entry['title'].lower()) or (kw in entry['abstract'].lower()):
                # print(kw)
                # print(entry['title'])
                # print(entry['abstract'])
                keyword_counter[kw] += 1
                count += 1
                found = True
                break
        matches.append(found)
    print(keyword_counter)
    print(f"Matches found: {count}")
    return matches

def entries_to_df(entries):
    # Extract title, author, year, venue, doi and put them in a pandas dataframe.
    data = []

    fields = ['index','title','abstract','author', 'year', 'venue', 'doi']
    for i, entry in enumerate(entries):
        # Determine venue based on series or journal
        venue = ''
        if 'series' in entry:
            venue = entry['series']
        elif 'journal' in entry:
            venue = entry['journal']

        # Extract other fields
        row = [
            i,
            entry['title'],
            entry['abstract'],
            entry['author'], 
            entry['year'], 
            venue,
            entry['doi']
        ]
        data.append(row)
    df = pd.DataFrame(data, columns=fields)
    return df

def create_bib(file, data, indices=[]):
    # Create new bib file based on entries
    # file: path of new file (str)
    # data: list of bib entries to be converted to bib file
    # indices (optional): indices of data to be used
    
    # Remove abstract since it can cause issues in latex
    for entry in entries:
        entry.pop("abstract", None)
    
    if (not indices.empty):
        output = bibtexparser.Library(list(np.array(data)[indices]))
    else:
        output = bibtexparser.Library(data)
    #bibtexparser.write_file(file, output)
    bib = bibtexparser.write_string(output)
    with open(file,'wb') as f:
        f.write(bib.encode('utf8'))
    print(f"{len(output.blocks)} entries written to file {file}")
    
def create_pie_plot(x):
    # x: the values and value counts for a df column
    cmap = plt.colormaps["Pastel1"]
    percent = (100*x)/(x.sum())
    #labels = ['{0}\n{1:1.1f} %'.format(i,j) for i,j in zip(llm_types.index, percent)]
    fig, ax = plt.subplots(figsize=(6,6))

    wedges, _ = ax.pie(
        x,
        colors=cmap.colors[:len(x)],
        wedgeprops={"edgecolor": "black"},
        startangle=90
    )

    # Add labels manually
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1  # mid-angle of wedge
        x_coord = np.cos(np.deg2rad(ang))
        y_coord = np.sin(np.deg2rad(ang))

        label = f"{x.index[i]}\n{percent.iloc[i]:.1f}%"

        if percent.iloc[i] >= 10:
            # Inside slice
            ax.text(
                0.6*x_coord, 0.6*y_coord, label,
                ha="center", va="center", fontsize=9
            )
        elif percent.iloc[i] >= 3:
            # Outside with arrow
            ax.annotate(
                label,
                xy=(x_coord, y_coord),           # wedge center
                xytext=(1.1*x_coord, 1.1*y_coord),  # label position outside
                ha="left", va="center", fontsize=8,
                arrowprops=dict(arrowstyle="-", connectionstyle=f"angle,angleA=0,angleB={ang}")
            )
        else:
            # Outside with arrow
            ax.annotate(
                label,
                xy=(x_coord, y_coord),           # wedge center
                xytext=(1.2*x_coord, 1.2*y_coord),  # label position outside
                ha="center", va="center", fontsize=8,
                arrowprops=dict(arrowstyle="-", connectionstyle=f"angle,angleA=0,angleB={ang}")
            )
    plt.show()
    
def create_bar_plot(x):
    # x: the values and value counts for a df column
    fig, ax = plt.subplots(figsize=(6,6))
    ax.bar(x.index,x.values)
    if (len(x.index) > 3):
        plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
    

def print_references(df, entries, col, oneline=False):
    """
    Get the reference keys from bib entries from the dataframe df for each class in specified column
    """
    refs = {}
    n = len(df[df[col].notna()])
    print(f"\n{col}")
    catn = {} # count for each option
    for key, group in df[df[col].notna()].groupby(col):
        #print(group["Index"])
        bib_keys = [entries[i].key for i in group["Index"]]
        refs[key] = "\\cite{" + ",".join(bib_keys) + "}"
        catn[key] = len(bib_keys)
    for ref in refs.keys():
        if oneline: # Write it like a latex row
            print(f"\\hspace{{3mm}} {ref} & {catn[ref]} ({(100*catn[ref])/n:.1f}\%) & {refs[ref]} \\\\")
        else:
            print(f"{catn[ref]} ({(100*catn[ref])/n:.1f}\%)")
            print(f"{ref}\n{refs[ref]}")


llm_keywords = ["llm","large language model","language model","code generation model","plm","pre-trained","pretrained","pre-training","pretraining","chatgpt","gpt","bert","t5","llama","bart"]

df_raw = entries_to_df(entries)
matches = keyword_search(entries,llm_keywords)
df_raw['LLM-related'] = matches

df_raw.to_excel('Output/raw_output.xlsx',index=False)

# Read the manually created excel sheet, with removed false positives and added information
df = pd.read_excel("../Results/MappingStudyLLM.xlsx","MappingStudyLLM")

# Create bib file for LLM related papers with downstream task
LLM_related = df[df["Downstream Task"] == True]
indices = LLM_related["Index"]
create_bib("Output/MappingStudy.bib",entries,indices)

# Create pie chart for LLM Task Type (classification, generation, recommendation or multiple)
llm_types = df["LLM Task Type"].value_counts().rename(index={"Classification & Recommendation":"Class. & Rec.","Generation & Classification":"Class. & Gen."})
create_bar_plot(llm_types)

# Get references for each LLM Task Type
print_references(df, entries, "LLM Task Type")

# Get all papers included in the mapping study (after inclusion/exclusion, etc.)
df_mapping = df[df['Full Access'] == "Y"]
# Full df_mapping references
print_references(df_mapping, entries, 'Full Access')

# Create pie chart for venues for classification papers
venues = df_mapping["Venue"].value_counts().rename(index={"ICSE '24":"ICSE"," IEEE Transactions on Software Engineering ":"TSE","FSE 2024":"FSE"})
create_bar_plot(venues)

# References
print_references(df_mapping, entries, "Venue")


# Crate pie chart for prompt usage 
prompts = df_mapping["Uses prompts"].value_counts().rename(index={"N":"No","Y":"Yes"})
create_bar_plot(prompts)
# References
print_references(df_mapping, entries, "Uses prompts")

# Go through all the evaluation metrics and other info:
for metric in ["Precision","Recall","Accuracy","F-Score","AUC","MCC","Calibration metric","Fairness metric", "Robustness metric", "Robustness test (adversarial, data distribution, noise)","Confusion matrix", "ROC Plot", "Evaluation results for multiple datasets","Baseline (ext, own, no)","Metrics justification (yes, no, implicit, previous work)","Stat. sign (between approaches)","Sign. test justification"]:
    print_references(df_mapping, entries, metric)

# Go through prompt-related information:
df_prompts = df_mapping[df_mapping["Uses prompts"] == "Y"]
for metric in ["Prompt Approach","Prompt Pattern","Prompt approach just.","Prompt design just.", "Prompt Comparison","Temperature"]:
    print_references(df_prompts,entries,metric,oneline=True)












