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

    **  In our experiments, we choose to store all the output data in a list called output and write it in a file
        just after the process finished. This may use a lot of memory, so if you intend to reproduce our experiments,
        you may want to modify this process to safe in file after a limited number of interactions.

"""

import model
import time
import copy
import numpy as np


def cluster(S, delta, beta, mi, eps, bounded_delta, output_path):
    """

    :param S: Input Stream
    :param delta: maximum time allowed for a tuple to be in memory.
    :param beta: maximum number of clusters in memory at once
    :param mi: number of clusters kept in memory to calculate 'tau'
    :param eps: parameter of Differential Privacy
    :param bounded_delta: sensitivity from Differential Privacy
    :param output_path: file path to write the output stream
    """

    init = time.time()

    # this list will store all output tuples. One may want to modify it to save in disk after a few iterations.
    output = []

    # Creating lists of anonymized and non-anonymized clusters:
    non_anonymized_clusters = []
    # domain of each attribute in the stream
    qid_atts_domain = model.QidAttsDomain()
    # tau:  if pushing a tuple_ to a cluster makes the information loss greater than tau, generate new cluster.
    tau = model.Tau(qid_atts_domain, mi, 0)

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
        offset = pid

        # create Tuple object with msg received.
        tuple_ = model.Tuple(pid, pid, qid_atts)
        # update general min, max for qid_atts in tuple_
        qid_atts_domain.put_values(tuple_.qidAtts)

        # selects the best cluster to add the new tuple.
        # if cluster==None, a new cluster must be created
        cluster = best_selection(non_anonymized_clusters, tuple_, tau, beta, qid_atts_domain)

        if cluster is None:
            new_cluster = model.Cluster(tuple_)
            non_anonymized_clusters.append(new_cluster)
        else:
            cluster.add_tuple(tuple_)

        # Output expiring_tuple
        # get_expiring_tuple returns (tuple_cluster)
        expiring_tuple_cluster = get_expiring_tuple(non_anonymized_clusters, offset, delta)

        if expiring_tuple_cluster is not None:
            delay_constraint(expiring_tuple_cluster, tau,
                             non_anonymized_clusters,
                             eps, bounded_delta, output)

    # since our stream might not be infinite, we may need to output the rest of the stream
    # that is, tuples that may not have expired yet.
    for cluster_ in list(non_anonymized_clusters):
        output_cluster(cluster_, non_anonymized_clusters,
                       tau, eps, bounded_delta, output)

    print("WRITING FILE:")
    print("PATH: {}".format(output_path))
    np.savetxt(output_path + "/result.csv", output, delimiter=",", fmt='%f')
    print("--------------\n")
    print("--- %s seconds ---" % (time.time() - init))


def get_expiring_tuple(non_anonymized_clusters, t, delta):
    """
    Returns expiring tuple as (t,C) where C is the cluster t belongs to.
    :param non_anonymized_clusters: a list of the non_anonymized_clusters
    :param t: current time
    :param delta: delay constraint
    :return: (tuple, cluster) -> expiring tuple and its cluster
    """
    for cluster in non_anonymized_clusters:
        for tuple_ in cluster.tuples:
            if tuple_.offset == t - delta:
                expiring_tuple = (tuple_, cluster)
                break
        else:  # if didn't find expiring tuple
            continue
        break  # if expiring tuple was found
    try:
        expiring_tuple
    except NameError:
        expiring_tuple = None
    return expiring_tuple


def best_selection(non_anonymized_clusters, tuple_, tau, beta, qid_atts_domain):
    """
    Returns the best cluster in non_anonymized_clusters if one exists, or None if a new Cluster should be created

    :param non_anonymized_clusters: a list of the non_anonymized_clusters
    :param tuple_: incoming tuple_ that needs a new Cluster
    :param tau: info_loss limit of a cluster
    :param beta: maximum number of non-ks-anonymized clusters
    :param qid_atts_domain: domain of each attribute in the stream
    :return: best cluster to add incoming tuple_, or None if a new cluster should be created
    """
    # Calculate enlargement of all clusters in case they receive the new tuple_.

    # start_time = time.clock()
    if non_anonymized_clusters:
        # a list of pairs [(cluster, enlargement)]
        enlargement_set = [(cluster, enlargement(qid_atts_domain, cluster, tuple_))
                           for cluster in non_anonymized_clusters]
        # a list of clusters with minimum enlargement
        set_min_clusters = [cluster[0] for cluster in enlargement_set
                            if cluster[1] == min(enlargement_set, key=lambda t: t[1])[1]]

        # check which clusters have info_loss not greater than tau
        set_clusters_ok = []
        for cluster in set_min_clusters:
            aux_cluster = copy.deepcopy(cluster)
            aux_cluster.add_tuple(tuple_)
            cluster_info_loss = info_loss(aux_cluster, qid_atts_domain)

            if cluster_info_loss <= tau.value:
                set_clusters_ok.append(cluster)

        # if no Cluster has info_loss less or equal to tau, must check if a new cluster could be created
        if not set_clusters_ok:
            # if |non_anonymized_clusters| >= beta, a new cluster cannot be created
            if len(non_anonymized_clusters) >= beta:
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


def info_loss(cluster, qid_atts_domain):
    """
    Returns the information loss of a cluster

    :param cluster: cluster to calculate info_loss
    :param qid_atts_domain: domain of each attribute
    :return: info_loss of cluster
    """
    total_info_loss = 0
    for genAtt in cluster.genAtts:
        total_info_loss += att_info_loss(cluster.genAtts[genAtt], qid_atts_domain.qidAtts[genAtt])
    return total_info_loss / len(cluster.genAtts)


def att_info_loss(domain, general_domain):
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


def enlargement(qid_atts_domain, cluster, tuple_or_cluster):
    """
    Calculates difference of info_loss of cluster with and without tuple_(or other cluster)

    :param qid_atts_domain: domain of each attribute
    :param cluster: cluster to add tuple_(or other cluster) to calculate impact of it
    :param tuple_or_cluster: tuple_ or cluster to be added to calculate impact of it
    :return: info_loss(cluster+tuple_or_cluster) - info_loss(cluster)
    """

    if isinstance(tuple_or_cluster, model.Tuple):
        # generating a new cluster containing the old cluster + tuple_
        cluster_ = copy.deepcopy(cluster)
        cluster_.add_tuple(tuple_or_cluster)

        # calculating infoLos for both clusters
        cluster_info_loss = info_loss(cluster, qid_atts_domain)
        cluster_prime_info_loss = info_loss(cluster_, qid_atts_domain)
        return cluster_prime_info_loss - cluster_info_loss

    else:
        raise ValueError("'tuple_or_cluster' must be Tuple")


def delay_constraint(tuple_cluster, tau, non_anonymized_clusters,
                     eps, bounded_delta, output):
    """
    Calls output_cluster

    :param tuple_cluster: (tuple_, cluster) pair, where tuple_ belongs to cluster and tuple_ is expiring.
    :param tau: average info_loss from mi latest clusters outputed
    :param non_anonymized_clusters: a list of non anonymized clusters
    :param eps: parameter from DP
    :param bounded_delta: sensitivity from DP
    :param output: list of output stream (so it is not saved after each iteration)
            WARNING: THIS MIGHT USE A LOT OF MEMORY. YOU MAY PREFER TO MODIFY IT TO SAVE AFTER A NUMBER OF ITERATIONS!
    """
    cluster = tuple_cluster[1]

    output_cluster(cluster, non_anonymized_clusters, tau, eps, bounded_delta, output)


def suppress_cluster(cluster):
    """
    Gives to each tuple_ from cluster the cluster's generalization
    :return: None
    """
    centroid = cluster.centroid()
    for tuple_ in cluster.tuples:
        tuple_.qidAtts['centroid'] = centroid


def output_cluster(cluster, non_anonymized_clusters, tau, eps, bounded_delta, output):
    """
    Output each tuple from cluster and the remove cluster from non_anonymized_clusters.
    Update tau with info_loss of the output cluster,
    """

    suppress_cluster(cluster)
    tau.update(cluster)

    cluster_noise = model.noise(bounded_delta, len(cluster), eps)

    for tuple_ in cluster.tuples:
        centroid = tuple_.qidAtts['centroid'][0]
        register_noise = model.noise(bounded_delta, 1, eps)
        output.append([cluster.id, tuple_.pid, tuple_.qidAtts['att'], centroid,
                       cluster_noise, centroid+cluster_noise, len(cluster),
                       register_noise, tuple_.qidAtts['att'] + register_noise])

    non_anonymized_clusters.remove(cluster)
