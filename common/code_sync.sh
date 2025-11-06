# Dell端运行，会在目的路径下创建文件夹
rsync -avzP --delete /home/dell/zzh/openpi/common airobot@192.168.50.28:/home/airobot/zzh/openpi #同步dell代码到arm
rsync -avzP --delete /home/dell/zzh/openpi/src airobot@192.168.50.28:/home/airobot/zzh/openpi
rsync -avzP --delete /home/dell/zzh/openpi/data airobot@192.168.50.28:/home/airobot/zzh/openpi

# rsync -avzP --delete dell@192.168.50.186:/home/dell/zzh/openpi/src /home/airobot/zzh/openpi #拉取arm代码到dell
# rsync -avzP --delete dell@192.168.50.186:/home/dell/zzh/openpi/common /home/airobot/zzh/openpi


# Arm端
# rsync -avzP --delete /home/airobot/zzh/openpi/common dell@192.168.50.186:/home/dell/zzh/openpi #同步arm代码到dell，在目的路径下创建文件夹
# rsync -avzP --delete /home/airobot/zzh/openpi/src dell@192.168.50.186:/home/dell/zzh/openpi

# rsync -avzP --delete dell@192.168.50.186:/home/dell/zzh/openpi/data /home/airobot/zzh/openpi  #同步dell代码到本地arm，源在目的路径下创建文件夹
# rsync -avzP --delete dell@192.168.50.186:/home/dell/zzh/openpi/common /home/airobot/zzh/openpi
# rsync -avzP --delete dell@192.168.50.186:/home/dell/zzh/openpi/src /home/airobot/zzh/openpi