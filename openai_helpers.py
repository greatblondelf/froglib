import openai
import json
import numpy as np
import time
import yaml


#use this in your code:
# with open("/home/frog/credential_file.yml", 'r') as ymlfile:
#    cfg = yaml.safe_load(ymlfile)
#    our_api_key = cfg['creds']['chatgpt_key']


class LLMEmbedder:
    def __init__(self):
        self.api_key = None

    def set_up_embedder(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def llm_embed(self, prompt_text):
        clustered_objects = 'job titles'

        # do the prompt text replacment (expansions)
        prompt_text_expanded = self.expand_text_abbreviations(prompt_text)

        output = None
        while output is None:
            time.sleep(0.4) # avoid hitting rate limiter
            #remember 46,000 seconds it about 12 hours, and 
            # the "rate limit" stated on the OpenAI site is 200 tokens per minute
            #https://platform.openai.com/docs/guides/rate-limits/overview
            try:
                response = openai.Embedding.create(
                    input='What is a ' + prompt_text_expanded + ', in the context of ' + clustered_objects + '?',
                    model="text-embedding-ada-002"
                )
                output = response['data'][0]['embedding']
            except Exception as e:
                print('Error in API Call, retrying API embedding query: ')
                print(e)
        return output
    
    def llm_response(self, prompt_text):
        ret_text = None
        while ret_text is None:
            time.sleep(0.4) # avoid hitting rate limiter
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                    {"role": "user", "content":prompt_text}
                    ]
                )
                ret_text = completion['choices'][0]['message']['content'].replace('\n','')
            except Exception as e:
                print('Error in API Call, retrying API embedding query: ')
        return ret_text # , completion


# Use ClustersByExample like so:
# cluster_defs_original =[
#   {
#     "display_name": "ABM - Marketing - Contacts Only",
#     "seed_titles": ["VP Marketing", "CMO", "Chief Marketing Officer", "Director Demand Generation", "VP Demand Generation", "Marketing Specialist", "Digital Marketing Manager", "Marketing Coordinator"]
#   },
#   {
#     ...
#   }
# ]
# cbe = ClustersByExample()
# cbe.add_api_key(our_api_key)
# # build up a set of clusters:
# cluster_defs_json_orig = {}
# for item in cluster_defs_original:
#     cluster_defs_json_orig[item['display_name']] = item['seed_titles']
# cbe.add_existing_clusters(cluster_defs_json)
# embedded_clusters = cbe.embed_clusters()
# # then embed other inputs and use a distance metric to cluster them

class ClustersByExample:
    def __init__(self):
        self.cluster_defs = {}
        self.embedder = None
        self.api_key = None

    def add_api_key(self, api_key):
        self.api_key = api_key
        self.embedder = LLMEmbedder()
        self.embedder.set_up_embedder(api_key=self.api_key)

    def add_existing_clusters(self, clusters):
        self.cluster_defs.update(clusters)
        return self.cluster_defs

    def embed_clusters(self):
        self.embedded_clusters = {}
        for cluster_header, example_names in self.cluster_defs.items():
            embeddings = np.array([self.embedder.embed(i) for i in example_names])
            mean = np.mean(embeddings, axis=0)
            std_dev = np.std(embeddings, axis=0)

            self.embedded_clusters[cluster_header] = {"mean": mean.tolist(), "std_dev": std_dev.tolist()}
        return self.embedded_clusters

    def get_embedded_clusters(self):
        return self.embedded_clusters


class ClusterComparison:
    def __init__(self, distance_threshold, embedded_clusters=None):
        self.distance_threshold = distance_threshold
        self.embedded_clusters = embedded_clusters

    def add_api_key(self, api_key):
        self.api_key = api_key
        self.embedder = Embedder()
        self.embedder.set_up_embedder(api_key=self.api_key)

    def set_job_clusters(self, embedded_clusters):
        self.embedded_clusters = embedded_clusters

    def get_job_clusters(self):
        return self.embedded_clusters

    def assign_clusters(self, job_title, scale_dist = False ,job_embedding=None):
        if job_embedding is None:
            embeddings = self.embedder.embed(job_title)
            job_embedding = embeddings
        assignments = []
        distances = []
        
        for cluster_header, cluster_embeddings in self.embedded_clusters.items():
            #NOTE:  We keep the standard deviations as well, in case we hav enough titles to successfully do distance scaling:
            if scale_dist:
                dist = np.linalg.norm((job_embedding - (np.array(cluster_embeddings["mean"])))/np.array(cluster_embeddings["std_dev"]) )
            else:
                dist = np.linalg.norm((job_embedding - (np.array(cluster_embeddings["mean"]))))

            print('---')
            print(f"CLUSTER HEADER: {cluster_header}")
            print(f"JOB TITLE: {job_title}")
            print(f"DISTANCE: {dist}")
            if dist <= self.distance_threshold:
                assignments.append(cluster_header)
                distances.append(dist)
        #arrange the clusters in order of distance
        ret = [x for _, x in sorted(zip(distances, assignments))]
        return ret # assignments