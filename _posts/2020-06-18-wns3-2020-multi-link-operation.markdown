---
layout: post
title:  "Workshop on ns-3 2020 - IEEE 802.11be Multi-link Operation in ns-3"
date:   2020-06-18 12:00:00 -0600
comments: true
---

Lightning talk presented at [Workshop on ns-3 2020](https://www.nsnam.org/research/wns3/wns3-2020/program/) virtual conference. ns-3 is an open source software platform for modeling computer networks including Wi-Fi and conducting performance evaluation studies. This talk introduces 802.11be Multi-link Operation to ns-3 audience and discusses potential extensions to ns-3 Wi-Fi module. 

## Slides

[Download slides][slides]

<p align = "center">
<iframe src="https://docs.google.com/viewer?url=https://github.com/sharan-naribole/sharan-naribole.github.io/raw/master/pdfs/wns3_2020_mlo_lightning_talk.pdf&embedded=true" width="100%" height="600px" style="border:thick solid #708090 ;">Your browser does not support the PDF embedding. </iframe>
</p>


### Slide 1: 802.11be Multi-link Operation (MLO)

IEEE 802.11be represents the next-generation Wi-Fi standard beyond the capabilities of 802.11ax products which are now being deployed. Concurrently, there has been an emergence of 802.11 devices
with multiple radios, capable of operating simultaneously on multiple channels possibly distributed over 2.4 GHz, 5 GHz and 6 GHz. To take advantage of the
multi-radio devices, MLO represents the framework to enable packet-level aggregation at the MAC layer. This means frames from a single 
traffic session e.g. video can be sent on multiple links using the first available link. Link is the general term defined in 802.11 standard for a unique wireless channel. Being able to send data from a traffic session using the first available channel
among multiple channels has potential to improve throughput and reduce latency.

### Slide 2: Multi-link Device (MLD) Architecture

Let's take a closer look at how the architecture would look like. We refer to 802.11 device capable of MLO as a Multi-link Device (MLD).
MLD is a logical entity with a single MAC-SAP interface to the upper layers. This means the upper layers do not need to know how many links the MLD is operating on.
Within the MLD, there can be one or more STAs where STA represents a MAC-PHY instance operating on a link. To make the operation efficient, the authentication and asssociation 
states are maintained at the MLD level. This means the end user device does not need to establish connnections separately on each link and can perform a single setup for multiple links.
Similarly, the Block ACK agreement and sequence number per TID is established at MLD level to enable packet aggregation on multiple links.

### Slide 3: Medium Access in MLO

By default, the medium access will be independent on each link with no coordination/medium state information sharing required between the STAs of an MLD. However, when two links are operating on nearby channels, the MLD might not have sufficient RF isolation and lack the ability to receive frames on one link while simultaneously transmitting on the other link. We refer to this ability as STR (simultaneous transmit-receive) capability. Typically, AP devices are many-antenna systems and the AP establishes the channels of operation. Therefore, it is reasonable to assume that AP maintains STR capability always on all pairs of links. In contrast, a non-AP MLD might lack STR capability for certain channel combinations due to smaller form factor and simpler design compared to AP. When the non-AP MLD lacks STR capability, if medium state information is not shared between its STAs, it can lead to data reception failure as illustrated in the top figure. Therefore, some level coordination is required between the STAs. 

### Slide 4: MLD Architecture in ns-3: MLD level

Bringing the above concepts together, if we envision a Node in ns-3, it is still going to have a single WifiNetDevice installed but now the same WifiNetDevice will be connected to multiple channels representing the links of operation. Extending the ns-3 Wi-Fi architecure for 802.11be involves careful consideration of functional distributions of operations at the MLD level and operations at individual STA level. For example, the MacMiddle class might contain the sequence control and reordering of frames received over multiple links. Similarly, the TxOp class which currently handles the queueing per Access Category needs to be extended to multiple links so that frames of a traffic session can be transmitted on the first available link.

### Slide 5: MLD Architecture in ns-3: STA level

In contrast, a few other functional blocks can continue to exist at the STA level. For example, Remote Station Manager which connects link adaptation algorithms and TX vector generation has functions specific to a link. This is because supported rates and rate adaptation of a link are not dependent on other link's conditions and operation parameters. Similarly, MacLow responsible for frame exchanges and interfacing with PHY is separate for each STA. As previously highlighted, for non-STR MLDs, medium state information sharing may be required between the STAs. Therefore, we propose the ChannelAccessManager block to be common for all the STAs of an MLD with Listeners from WifiPhy and MacLow of each STA to utilize real-time CCA and NAV information of each link.

### Conclusion

In conclusion, MLO, the key feature of next-generation IEEE 802.11be standard project involves a giant leap in architecture, protocol design and modeling in ns-3.

[slides]: https://github.com/sharan-naribole/sharan-naribole.github.io/raw/master/pdfs/wns3_2020_mlo_lightning_talk.pdf