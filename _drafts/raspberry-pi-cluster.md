
http://www.mccarroll.net/blog/rpi_cluster2/index.html
https://www.raspberrypi.org/documentation/installation/installing-images/mac.md
https://www.youtube.com/watch?v=LgdBaARF1Q8&list=PLQVvvaa0QuDf9IW-fe6No8SCw-aVnCfRi

# /boot/config.txt
hdmi_force_hotplug=1
hdmi+group=2
hdmi_mode=1
hdmi_mode=87
hdmi_cvt=800 480 60 6 0 0 0

There have been some changes since this was first published. First, I'm using Raspberry Pi 3's which have their own wifi. I'm using the same basic set up as here but, obviously, you don't need the dongle. Even with wifi, I want the workers to be wired, not wireless so I still wanted this basic setup. The changes are as follows. They must be done for each worker.

First, you must also edit the /etc/dhcpcd.conf file. Add the following to the end where XXX is the worker's IP address.

# Custom static IP address for eth0.
interface eth0
static ip_address=192.168.1.XXX/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1

Second, if you need a name server, you should modify /etc/resolv.conf to something like:
nameserver 8.8.8.8

Third, and I don't know if this was strictly necessary but it was often useful to get back to the master from any worker if I happened to be logged in directly so I added to /etc/hosts:
192.168.1.1 pi0

The killer is not modifying the dhcpcd.conf file. You will get the IP address for unsuccessful DHCP and wonder WTH?

pi@rpy02:~ $ sudo vi /etc/hostname
pi@rpy02:~ $ sudo vi /etc/hosts
pi@rpy02:~ $ sudo vi /etc/network/interfaces
pi@rpy02:~ $ sudo vi /etc/dhcpcd.conf
