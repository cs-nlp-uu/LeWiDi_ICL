import json


def load_data (file_path, is_varierrnli=False, is_test=False):
    
    """
    INPUT VARIABLES:
    - file_path: file path to the JSON data in your Google Drive
    - is_varierrnli: optional, set it to 1 if you are evaluation the VariErrNLI dataset. Otherwise don't pass it to the function (default is None).


    OUTPUT VARIABLES:

    - targets_soft:
        Type: list of lists (Csc, MP, Par), list of lists of lists (VariErrNLI)
        Content: a list containg a list for each item flat integer labels representing the true soft labels distribution (i.e., the "targets"), a value for each soft label. In the case of VariErrNLI is further nested into 3 other lists (for Contradiction, Entailment, Neutral)
        Details:
        targets_soft_mp  = [[0.4,0.6], [...] ,[0.25,0.75]]
                ie: [[item1_SoftLabel"0", item1_SoftLabel"1"], ... ,[item-n_SoftLabel"0",item-n_SoftLabel"1"]]

        targets_soft_par = [[0.0, 0.25, 0.25, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0], ... ,[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]]
                ie: [[item1_SL"-5", item1_SL"-4", item1_SL"-3", item1_SL"-2", item1_SL"-1", item1_SL"0", item1_SL"1", item1_SL"2", item1_SL"3", item1_SL"4", item1_SL"5"],[...] ,[item-n_SL"-5", item-n_SL"-4", item-n_SL"-3", item-n_SL"-2", item-n_SL"-1", item-n_SL"0", item-n_SL"1", item-n_SL"2", item-n_SL"3", item-n_SL"4", item-n_SL"5"]]

        targets_soft_csc = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], ... ,[0.0, 0.0, 0.5, 0.16666666666666666, 0.3333333333333333, 0.0, 0.0]]
                ie: [[item1_SL"0", item1_SL"1", item1_SL"2", item1_SL"3", item1_SL"4", item1_SL"5", item1_SL"6"], ... ,[item-n_SL"0", item-n_SL"1", item-n_SL"2", item-n_SL"3", item-n_SL"4", item-n_SL"5", item-n_SL"6"]]

        targets_soft_ven = [[[1.0, 0.0], [0.5, 0.5], [0.25, 0.75]], ... ,[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]
                ie: [[[item1_Contradiction_SoftLabel"0", item1_Contradiction_SoftLabel"1"],[item1_Entailment_SoftLabel"0", item1_Entailment_SoftLabel"1"],[item1_Neutral_SoftLabel"0", item1_Neutral_SoftLabel"1"]], ... ,[[item-n_Contradiction_SoftLabel"0", item-n_Contradiction_SoftLabel"1"],[item-n_Entailment_SoftLabel"0", item-n_Entailment_SoftLabel"1"],[item-n_Neutral_SoftLabel"0", item-n_Neutral_SoftLabel"1"]]]

    - targets_pe:
        Type: list of lists (Csc, MP, Par), list of lists of lists (VariErrNLI)
        Content: Each item is a flat list of integer labels (e.g., [0, 1, 2]) representing the annotation from each annotator (specified in annotators_pe) for the specific item.
        Details:
        targets_pe_csc = [[1,1,1,1],...,[2,4,2,2,3,4]]
        targets_pe_mp  = [[0,1,1,1,0],...,[1,1,1,0]]
        targets_pe_par = [[-4,-3,0,-1],...,[5,4,4,5]]
        targets_pe_ven = [[[0,0,0,0],[0,1,0,1], [1,1,1,0]],...,[[0,0,0,0],[1,1,1,1],[0,0,0,0]]]
        Note:
        csc, mp, par: [[Annotation for item1 by first annotator of item1, ..., Annotation for item1 by n-annotator of the item 1 ],...,[Annotation for item-n by first annotator of item-n, ... , Annotation for item-n by n-annotator of item-n ]]
        vari_err_nli: [[[Annotation for contradiction for item1 by first annotator of item1, ..., Annotation for contradiction for item1 by n-annotator of the item 1 ],[Annotation for entilment for item1 by first annotator of item1, ..., Annotation for entilment for item1 by n-annotator of the item 1 ],[Annotation for neutral for item1 by first annotator of item1, ..., Annotation for neutral for item1 by n-annotator of the item 1 ]],...,[[Annotation for contradiction for item-n by first annotator of item-n, ..., Annotation for contradiction for item-n by n-annotator of the item 1 ],[Annotation for entilment for item-n by first annotator of item-n, ..., Annotation for entilment for item-n by n-annotator of the item 1 ],[Annotation for neutral for item-n by first annotator of item-n, ..., Annotation for neutral for item-n by n-annotator of the item 1 ]]]

    - annotators_pe:
        Type: list of strings
        Contents: List of annotator ID strings for each data item.
        Details: They are extracted directly from the "annotators" field in the JSON data.
        annotators_pe_csc = ['Ann844,Ann845,Ann846,Ann847', ...,'Ann60,Ann61,Ann62,Ann63,Ann64,Ann65']
        annotators_pe_mp  = ['Ann0,Ann20,Ann59,Ann62,Ann63',...,'Ann499,Ann500,Ann501,Ann505']
        annotators_pe_par = ['Ann1,Ann2,Ann3,Ann4',...,'Ann1,Ann2,Ann3,Ann4']
        annotators_pe_ven = ['Ann1,Ann2,Ann3,Ann4',...,'Ann1,Ann2,Ann3,Ann4']

    - ids:
        Type: list
        Contents: Unique IDs of each example in the dataset.
        Details:These are the keys from the JSON data (e.g., "123", "456").

    - data
        Type: dict
        Contents: The full parsed JSON content from the input file.

    """
    
    # read json data
    with open(file_path, "r") as f:
        data = json.load(f)

    # initialize output variabiles
    annotators_pe = [item["annotators"].split(",") for _, item in data.items()]
    ids = list(data.keys())
    targets_soft = list()
    targets_pe = list()

    if is_test:
        return (targets_soft, targets_pe, annotators_pe, ids, data)

    # loop on each item
    for id_, content in data.items():
        # extract soft label of the item and append it to the targets' soft list
        soft_label = content.get("soft_label", {})
        soft_list = list(soft_label.values())

        # extract annotators and annotations of the item and append it to the targets' PE list

        if not is_varierrnli: # if the dataset is not varierrnli, just extract the annotations

            targets_soft.append(soft_list)

            annotation_dict = content["annotations"]
            annotations = [int(annotation_dict[ann]) for ann in content["annotators"].split(",")]
            targets_pe.append(annotations)



        else: # if the dataset is varierrnli loop on contradiction, entailment, neutral
            soft_list=[[v['0'], v['1']] for v in soft_label.values()]
            annotations = content.get("annotations", {})
            annotators = list(annotations.keys())
            num_annotators = len(annotators)

            label_to_index = ["contradiction", "entailment", "neutral"]
            # Initialize label vectors for each annotator
            label_vectors = {label: [0] * num_annotators for label in label_to_index}
            annotator_to_index = {annotator: idx for idx, annotator in enumerate(annotators)}

            # Fill in the label vectors
            for annotator, annotation_str in annotations.items():
                idx = annotator_to_index[annotator]
                for label in annotation_str.split(','):
                    label = label.strip()
                    if label in label_vectors:
                        label_vectors[label][idx] = 1

            targets_soft.append(soft_list)
            targets_pe.append(list(label_vectors.values()))

    return (targets_soft,targets_pe,annotators_pe,ids,data)


def load_all_data(config, project_root):
    """
    Load all datasets specified in the config file.
    input:
        - config: config file
        - project_root: project root
    output:
        - data: data for all datasets
    """
    for dataset_name in config['dataset_names']:
        for key, value in config['data'][dataset_name].items():
            config['data'][dataset_name][key] = value.replace('${PROJECT_ROOT}', project_root)

    data = dict()

    for dataset_name in config['dataset_names']:
        data[dataset_name] = dict()
        for split in ["train", "dev", "test"]:
            is_varierrnli = (dataset_name == "VariErrNLI")
            is_test = (split == "test")
            file_path = config['data'][dataset_name][f"{split}_file"]

            soft_labels, perspectivism, annotators_per_entry, ids, full_data = load_data(file_path, is_varierrnli=is_varierrnli, is_test=is_test)

            data[dataset_name][split] = {
                "soft_labels": soft_labels,
                "perspectivism": perspectivism,
                "annotators_per_entry": annotators_per_entry,
                "ids": ids,
                "data": full_data
            }

            print(f"Loaded {dataset_name} {split} data with {len(full_data)} examples.")
    
    return data

def load_prompt_template(config, file_path):
    """
    Load the prompt template from the config file.
    input:
        - config: config file
        - file_path: file path to the prompt template
    output:
        - data: prompt template
    """
    data = json.load(open(file_path, 'r'))

    print(f"Load the prompt template: version {data['version']}")

    for dataset_name in config['dataset_names']:
        general = data["content"]
        dataset_specific = data["datasets"][dataset_name]
        introduction = general["introduction"].replace("${TASK_NAME}", dataset_specific["task_name"])
        task_description = general["task_description"].replace("${INPUT_FORMAT}", dataset_specific["input_format"]).replace("${RESPONSE_FORMAT}", dataset_specific["response_format"]).replace("${LABEL_EXPLANATION}", dataset_specific["label_explanation"])


        data["datasets"][dataset_name]["prompt_template"] = introduction + " " + task_description + " " + general["style"] + "\n" + general["examples"] + "\n" + general["input"]

        print(f"Propmt for task {dataset_name} is like:\n {data['datasets'][dataset_name]['prompt_template']}")

    return data