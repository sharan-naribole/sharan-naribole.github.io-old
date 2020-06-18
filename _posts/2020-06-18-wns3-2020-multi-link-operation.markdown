---
layout: post
title:  "Workshop on ns-3 2020 - IEEE 802.11be Multi-link Operation in ns-3"
date:   2020-06-18 12:00:00 -0600
comments: true
---

Lightning talk presented at Workshop on ns-3 2020 virtual conference. This talk introduces 802.11be multi-link operation 
in ns-3 and potential extensions to ns-3 Wi-Fi module.

## Slides

<p align = "center">
<iframe src="https://docs.google.com/viewer?url=https://github.com/sharan-naribole/sharan-naribole.github.io/raw/master/pdfs/wns3_2020_mlo_lightning_talk.pdf&embedded=true" width="100%" height="600px" style="border:thick solid #708090 ;">Your browser does not support the PDF embedding. </iframe>
</p>

### Slide 1: 802.11be Multi-link Operation

IEEE 802.11be represents the next-generation Wi-Fi standard beyond the capabilities of 802.11ax products which are now being deployed. Concurrently, there has been an emergence of 802.11 devices
with multiple radios, capable of operating simultaneously on multiple channels possibly distributed over 2.4 GHz, 5 GHz and 6 GHz. To take advantage of the
multi-radio devices, Multi-link operation represents the framework to enable packet-level aggregation at the MAC layer. This means frames from a single 
traffic session e.g. video can be sent on multiple links using the first available link. Link is the general term defined in 802.11 standard for a unique wireless channel. Being able to send data from a traffic session using the first available channel
among multiple channels has potential to improve throughput and reduce latency.

### Slide 2: Multi-link Device (MLD) Architecture

Let's take a closer look at how the architecture would look like. We refer to 802.11 device capable of multi-link operation as a Multi-link Device (MLD).
MLD is a logical entity with a single MAC-SAP interface to the upper layers. This means the upper layers do not need to know how many links the MLD is operating on.
Within the MLD, there can be one or more STAs where STA represents a MAC-PHY instance operating on a link. To make the operation efficient, the authentication and asssociation 
states are maintained at the MLD level. This means the end user device does not need to establish connnections separately on each link and can perform a single setup for multiple links.
Similarly, the Block ACK agreement and sequence number per TID is established at MLD level to enable packet aggregation on multiple links.




