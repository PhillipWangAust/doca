"""
DOCA: Differential Privacy Online Clustering and Anonymization

IMPORTANT:
    Even though the model used in DOCA is able to deal with multiple attributes, the proposed solution, as published
    in the DPM workshop( https://link.springer.com/chapter/10.1007%2F978-3-030-00305-0_20) works in the context of
    a stream of a single attribute.

OUTPUT:
    The output generated from this process has the following attributes:
    [cluster_id, tuple_id, tuple_att_value, centroid, cluster_noise, centroid+cluster_noise,
    len(cluster), register_noise, tuple_att_value + register_noise]

    *   This output was generated just for experimental comparison.

    *   In our experiments, we choose to store all the output data in a list called output and write it in a file
        just after the process finished. This may use a lot of memory, so if you intend to reproduce our experiments,
        you may want to modify this process to safe in file after a limited number of interactions.

"""

import model
import time
import copy
import numpy as np


class Doca:
    def __init__(self, delta_time, beta, mi, budget, sensitivity):
        """
        :param delta_time: maximum time allowed for a tuple to be in memory.
        :param beta: maximum number of clusters in memory at once
        :param mi: number of clusters kept in memory to calculate 'tau'
        :param budget: parameter of Differential Privacy
        :param sensitivity: sensitivity from Differential Privacy
        :param cost_function: choose between error and info_loss to select a cluster for a tuple
         """
        self.delta_time = delta_time
        self.beta = beta
        self.mi = mi
        self.budget = budget
        self.sensitivity = sensitivity
        # Creating lists of anonymized and non-anonymized clusters:
        self.non_anonymized_clusters = []
        # domain of each attribute in the stream
        self.qid_atts_domain = model.QidAttsDomain()
        # tau:  if pushing a tuple_ to a cluster makes the information loss greater than tau, generate new cluster.
        self.tau = model.Tau(self.mi)
        self.current_time = 0
        self.noise_threshold = model.Tau(self.mi)

        # this list will store all output tuples. One may want to modify it to save in disk after a few iterations.
        self.output = []

    def cluster(self, S, output_path):
        """

        :param S: stream of tuples
        :param output_path: file path to write the output stream
        """

        init = time.time()

        for msg in S:
            '''
            For each new message in the input stream creates a new Tuple and add it to a cluster in non_anonymized_clusters,
            or creates a new cluster, if necessary
            After that, checks if exists an expiring tuple_ in any cluster in non_anonymized_clusters,
            if such tuple_ exists, calls delay_constraint in order to output it.
            '''

            # msg received as a list of [pid, {qid_atts}]
            pid = msg[0]
            qid_atts = msg[1]
            # since pid is a sequential number, it can be used as a timestamp
            self.current_time = pid

            # create Tuple object with msg received.
            tuple_ = model.Tuple(pid, pid, qid_atts)
            # update general min, max for qid_atts in tuple_
            self.qid_atts_domain.put_values(tuple_.qidAtts)

            # selects the best cluster to add the new tuple.
            # if cluster==None, a new cluster must be created
            cluster = self.best_selection(tuple_)

            if cluster is None:
                new_cluster = model.Cluster(tuple_)
                self.non_anonymized_clusters.append(new_cluster)
            else:
                cluster.add_tuple(tuple_)

            # Output expiring_tuple
            # get_expiring_tuple returns (tuple_cluster)
            _, expiring_cluster = self.get_expiring_tuple()

            if expiring_cluster is not None:
                self.output_cluster(expiring_cluster)

        # since our stream might not be infinite, we may need to output the rest of the stream
        # that is, tuples that may not have expired yet.
        for cluster_ in list(self.non_anonymized_clusters):
            self.output_cluster(cluster_)

        print("WRITING FILE:")
        print("PATH: {}".format(output_path))
        np.savetxt(output_path + "/result.csv", self.output, delimiter=",", fmt='%f')
        print("--------------\n")
        print("--- %s seconds ---" % (time.time() - init))

    def get_expiring_tuple(self):
        """
        Returns expiring tuple as (t,C) where C is the cluster t belongs to.
        :return: (tuple, cluster) -> expiring tuple and its cluster
        """
        expiring_tuple = None, None
        for cluster in self.non_anonymized_clusters:
            for tuple_ in cluster.tuples:
                if tuple_.offset == self.current_time - self.delta_time:
                    expiring_tuple = (tuple_, cluster)
                    break
            else:  # if didn't find expiring tuple
                continue
            break  # if expiring tuple was found
        return expiring_tuple

    def best_selection(self, tuple_):
        """
        Returns the best cluster in non_anonymized_clusters if one exists, or None if a new Cluster should be created

        :param tuple_: incoming tuple_ that needs a new Cluster
        :return: best cluster to add incoming tuple_, or None if a new cluster should be created
        """
        # Calculate enlargement of all clusters in case they receive the new tuple_.

        if self.non_anonymized_clusters:
            # a list of pairs [(cluster, enlargement)]
            enlargement_set = [(cluster, self.enlargement(cluster, tuple_))
                               for cluster in self.non_anonymized_clusters]
            # a list of clusters with minimum enlargement
            set_min_clusters = [cluster for cluster, enlargement in enlargement_set
                                if enlargement == min(enlargement_set, key=lambda t: t[1])[1]]

            # check which clusters have info_loss not greater than tau
            set_clusters_ok = []
            for cluster in set_min_clusters:
                aux_cluster = copy.deepcopy(cluster)
                aux_cluster.add_tuple(tuple_)
                cluster_info_loss = self.info_loss(aux_cluster)

                if cluster_info_loss <= self.tau.value:
                    set_clusters_ok.append(cluster)

            # if no Cluster has info_loss less or equal to tau, must check if a new cluster could be created
            if not set_clusters_ok:
                # if |non_anonymized_clusters| >= beta, a new cluster cannot be created
                if len(self.non_anonymized_clusters) >= self.beta:
                    # Must select from clusters with minimum enlargement one with minimum size
                    sizes = [len(cluster) for cluster in set_min_clusters]
                    index = sizes.index(min(sizes))
                    return set_min_clusters[index]
                else:
                    # A new cluster should be created in cluster()
                    return None
            else:
                # Select one cluster with minimum size from set_clusters_ok
                sizes = [len(cluster) for cluster in set_clusters_ok]
                index = sizes.index(min(sizes))
                return set_clusters_ok[index]
        else:
            # in case nonAnonymizedCluster is empty a new cluster should be created
            return None

    def info_loss(self, cluster):
        """
        Returns the information loss of a cluster

        :param cluster: cluster to calculate info_loss
        :param qid_atts_domain: domain of each attribute
        :return: info_loss of cluster
        """
        total_info_loss = 0
        for genAtt in cluster.genAtts:
            total_info_loss += self.att_info_loss(cluster.genAtts[genAtt], self.qid_atts_domain.qidAtts[genAtt])
        return total_info_loss / len(cluster.genAtts)

    def att_info_loss(self, domain, general_domain):
        """
        Calculates info_loss of each attribute

        :param domain: domain of attribute in cluster
        :param general_domain: general domain of attribute
        :return: info_loss of the attribute
        """
        minimum, maximum = domain
        minimum_general, maximum_general = general_domain
        try:
            loss = (maximum - minimum) / (maximum_general - minimum_general)
        except ZeroDivisionError:
            loss = 0
        return loss

    def enlargement(self, cluster, tuple_):
        """
        Calculates difference of info_loss of cluster with and without tuple_(or other cluster)

        :param cluster: cluster to add tuple_(or other cluster) to calculate impact of it
        :param tuple_: tuple_ or cluster to be added to calculate impact of it
        :return: info_loss(cluster+tuple_or_cluster) - info_loss(cluster)
        """

        if isinstance(tuple_, model.Tuple):
            # generating a new cluster containing the old cluster + tuple_
            cluster_ = copy.deepcopy(cluster)
            cluster_.add_tuple(tuple_)

            # calculating infoLos for both clusters
            cluster_info_loss = self.info_loss(cluster)
            cluster_prime_info_loss = self.info_loss(cluster_)
            return cluster_prime_info_loss - cluster_info_loss

        else:
            raise ValueError("'tuple_or_cluster' must be Tuple")

    def suppress_cluster(self, cluster):
        """
        Gives to each tuple_ from cluster the cluster's generalization
        :return: None
        """
        centroid = cluster.centroid()
        for tuple_ in cluster.tuples:
            tuple_.qidAtts['centroid'] = centroid

    def output_cluster(self, cluster):
        """
        Output each tuple from cluster and the remove cluster from non_anonymized_clusters.
        Update tau with info_loss of the output cluster,
        """

        self.suppress_cluster(cluster)

        self.tau.update(self.info_loss(cluster))
        cluster_noise = self.noise(len(cluster))

        self.noise_threshold.update(cluster_noise)

        for tuple_ in cluster.tuples:
            centroid = tuple_.qidAtts['centroid'][0]
            register_noise = self.noise(1)
            self.output.append([cluster.id, tuple_.pid, tuple_.qidAtts['att'], centroid,
                                cluster_noise, centroid+cluster_noise, len(cluster),
                                register_noise, tuple_.qidAtts['att'] + register_noise])

        self.non_anonymized_clusters.remove(cluster)

    def noise(self, cluster_size):
        """
        Returns a sample from Laplace distribution with mean=0 and scale=delta_c/cluster_size
        :param cluster_size: number of tuples in the cluster
        :return: a sample from Laplace distribution with mean=0 and scale=delta_c/cluster_size
        """
        delta_c = self.sensitivity / cluster_size
        scale = delta_c / self.budget
        return np.random.laplace(0, scale, 1)[0]
