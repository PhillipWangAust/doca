import numpy as np


class Tuple:
    id = 0

    def __init__(self, pid=0, offset=0, qid_atts=None):
        self.pid = Tuple.id
        Tuple.id += 1
        if qid_atts is None:
            qid_atts = {}
        self.qidAtts = qid_atts
        self.pid = pid
        self.offset = offset

    def __str__(self):
        return str(self.pid)


class Cluster:

    """
    tuples: list
        list of tuples in the cluster: [t1, t2, ...]
    genAtts: dict
        dictionary with minimum and maximum of each pid in Cluster that will be used in calculation of Tau.
        The key is the attribute's name, and the value is a a tuple (min, max) containing the minimum and maximum values
        in the cluster -> {att_name: (min, max)}.
    """

    id = 0

    def __init__(self, tuple_):
        self.id = Cluster.id
        Cluster.id += 1
        self.tuples = [tuple_]
        self.genAtts = {}
        # Transforming {key, value} in {key, (value, value)}
        for key, value in tuple_.qidAtts.items():
            self.genAtts[key] = (value, value)

    # add tuple to [tuples] and update min max in each attribute in genAtts
    def add_tuple(self, tuple_):
        """
        Adds a new tuple in the cluster and then calls put_values(tuple_.qidAtts).

        :param tuple_: new tuple
        """
        self.tuples.append(tuple_)
        self.put_values(tuple_.qidAtts)

    def put_values(self, qids):
        """
        Updates the genAtts list, i.e. the (min, max) of each attribute.
        :param qids: Attributes of a new tuple used to update genAtts.
        """
        for key in qids.keys():
            # if key in qidAtts, check if value < minimum or value > maximum.
            if key in self.genAtts:
                minimum, maximum = self.genAtts[key]

                if qids[key] < minimum:
                    minimum = qids[key]
                elif qids[key] > maximum:
                    maximum = qids[key]
                self.genAtts[key] = (minimum, maximum)
            else:
                self.genAtts[key] = (qids[key], qids[key])

    def centroid(self):
        """
        Calculates the cluster's centroid as the average of each attribute.

        :return: average of attributes from tuples in cluster.
        """
        sum_att = np.zeros(len(self.genAtts))
        for tuple_ in self.tuples:

            for i, att in enumerate(tuple_.qidAtts.values()):
                sum_att[i] += att

        mean_atts = sum_att/len(self.tuples)
        return mean_atts

    def __len__(self):
        """
        :return: number of tuples in the cluster.
        """
        return len(self.tuples)


# Setting min and max from all pids in all stream
class QidAttsDomain:
    """
    qidAtts: dict
        dictionary with minimum and maximum of each element in the stream so far that will be used in calculation of Tau
        The key is the attribute's name, and the value is a tuple (min, max) containing the minimum and maximum values
        -> {att_name: (min, max)}.
    """
    def __init__(self, qid_atts={}):
        self.qidAtts = {}
        # Transforming {key, value} in {key, (value, value)}
        for key, value in qid_atts.items():
            self.qidAtts[key] = (value, value)

    # consider qid = {qid,value}
    def put_values(self, qids):
        """
        Updates the genAtts list, i.e. the (min, max) of each attribute.
        :param qids: Attributes of a new tuple used to update genAtts.
        """
        for key in qids.keys():
            # if key in qidAtts, check if value < minimum or value > maximum.

            if key in self.qidAtts:
                minimum, maximum = self.qidAtts[key]
                if qids[key] < minimum:
                    minimum = qids[key]
                elif qids[key] > maximum:
                    maximum = qids[key]

                self.qidAtts[key] = (minimum, maximum)
            else:
                self.qidAtts[key] = (qids[key], qids[key])


class Tau:
    """
    Keeps track of the last mi cluster published, and calculates the average of their info_loss.
    """
    def __init__(self, mi=0, value=0):
        """
        :param mi: number of published clusters to be used to calculate Tau
        :param value: the info_loss average of the last mi published clusters
        """
        self.value = value
        self.last_clusters_info_loss = []
        self.mi = mi

    def update(self, cluster_info_loss):
        """
        Updates tau value.

        :param cluster_info_loss: info loss from last published cluster
        """

        if len(self.last_clusters_info_loss) < self.mi:
            self.last_clusters_info_loss.append(cluster_info_loss)
        # if anonymizedClusters size is >= mi, should pop the oldest one before adding
        else:
            self.last_clusters_info_loss.pop(0)
            self.last_clusters_info_loss.append(cluster_info_loss)

        self.value = sum(self.last_clusters_info_loss) / len(self.last_clusters_info_loss)



